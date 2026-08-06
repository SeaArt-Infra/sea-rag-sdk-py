from __future__ import annotations

import codecs
import json
import os
import socket
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib import error, parse, request

from .errors import APIError, RAGError
from .types import UploadFile

QueryParams = Mapping[str, Any]


class Transport:
    """Shared HTTP transport for SeaRAG resources."""

    def __init__(
        self,
        endpoint: str,
        api_key: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 180.0,
    ) -> None:
        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        self.endpoint = normalize_rag_endpoint(endpoint)
        self.api_key = api_key
        self.headers = dict(headers or {})
        self.timeout = timeout

    def get_json(self, path: str, query: QueryParams | None = None) -> Any:
        return self.request_json("GET", path, query=query)

    def post_json(self, path: str, body: Any = None) -> Any:
        return self.request_json("POST", path, body=body)

    def put_json(self, path: str, body: Any = None) -> Any:
        return self.request_json("PUT", path, body=body)

    def patch_json(self, path: str, body: Any = None) -> Any:
        return self.request_json("PATCH", path, body=body)

    def delete_json(self, path: str, body: Any = None) -> Any:
        return self.request_json("DELETE", path, body=body)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: QueryParams | None = None,
        body: Any = None,
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        raw = self._request_text(method, path, query, body, "application/json", headers)
        if raw == "":
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RAGError(f"expected JSON response, got: {_response_preview(raw)}") from exc
        _raise_rag_response_error(value)
        return value

    def post_multipart(
        self,
        path: str,
        files: Sequence[UploadFile],
        fields: Mapping[str, str] | None = None,
    ) -> Any:
        body, content_type = _encode_multipart(files, fields or {})
        raw = self._request_text(
            "POST",
            path,
            None,
            body,
            "application/json",
            {"Content-Type": content_type},
        )
        if raw == "":
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RAGError(f"expected JSON response, got: {_response_preview(raw)}") from exc
        _raise_rag_response_error(value)
        return value

    def post_stream(self, path: str, body: Any, on_chunk: Callable[[str], None]) -> None:
        if not callable(on_chunk):
            raise TypeError("on_chunk must be callable")
        req = self._build_request("POST", path, None, body, "text/event-stream", None)
        try:
            response = request.urlopen(req, timeout=None if self.timeout == 0 else self.timeout)
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise _http_error(exc.code, raw) from exc
        except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise RAGError(f"stream request failed: {exc}") from exc

        decoder = codecs.getincrementaldecoder("utf-8")()
        with response:
            while True:
                try:
                    raw = response.read(4096)
                except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                    raise RAGError(f"stream failed: {exc}") from exc
                if not raw:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        on_chunk(tail)
                    return
                chunk = decoder.decode(raw)
                if chunk:
                    on_chunk(chunk)

    def download(self, path: str) -> tuple[bytes, Mapping[str, str]]:
        req = self._build_request("GET", path, None, None, "*/*", None)
        try:
            with request.urlopen(req, timeout=None if self.timeout == 0 else self.timeout) as response:
                return response.read(), dict(response.headers.items())
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise _http_error(exc.code, raw) from exc
        except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise RAGError(f"request failed: {exc}") from exc

    def build_url(self, path: str, query: QueryParams | None = None) -> str:
        parsed = parse.urlsplit(self.endpoint)
        base_path = parsed.path if parsed.path.endswith("/") else f"{parsed.path}/"
        relative_path = path.lstrip("/")
        joined_path = _collapse_path(base_path + relative_path)

        values: dict[str, list[str]] = {}
        for key, value in parse.parse_qsl(parsed.query, keep_blank_values=True):
            values.setdefault(key, []).append(value)
        for key, value in (query or {}).items():
            if _is_zero_value(value):
                continue
            values[key] = _query_values(value)

        pairs = [(key, item) for key, items in values.items() for item in items]
        return parse.urlunsplit(parsed._replace(path=joined_path, query=parse.urlencode(pairs)))

    def build_headers(
        self,
        accept: str,
        has_body: bool,
        request_headers: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        if accept:
            headers["Accept"] = accept
        if has_body:
            headers["Content-Type"] = "application/json"
        headers.update({key: value for key, value in self.headers.items() if key.strip()})
        headers.update({key: value for key, value in (request_headers or {}).items() if key.strip()})
        if self.api_key and not _has_header(headers, "Authorization"):
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request_text(
        self,
        method: str,
        path: str,
        query: QueryParams | None,
        body: Any,
        accept: str,
        headers: Mapping[str, str] | None,
    ) -> str:
        req = self._build_request(method, path, query, body, accept, headers)
        try:
            with request.urlopen(req, timeout=None if self.timeout == 0 else self.timeout) as response:
                return response.read().decode("utf-8")
        except error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise _http_error(exc.code, raw) from exc
        except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise RAGError(f"request failed: {exc}") from exc

    def _build_request(
        self,
        method: str,
        path: str,
        query: QueryParams | None,
        body: Any,
        accept: str,
        headers: Mapping[str, str] | None,
    ) -> request.Request:
        payload: bytes | None
        if body is None:
            payload = None
        elif isinstance(body, bytes):
            payload = body
        else:
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self.build_url(path, query)
        if _is_debug_enabled():
            print(method, url, file=os.sys.stderr)
        return request.Request(
            url,
            data=payload,
            headers=self.build_headers(accept, payload is not None, headers),
            method=method,
        )


def normalize_rag_endpoint(endpoint: str) -> str:
    """Append the SeaArt RAG gateway route when it is absent."""
    if not isinstance(endpoint, str) or endpoint.strip() == "":
        return endpoint
    parsed = parse.urlsplit(endpoint)
    segments = [segment for segment in parsed.path.split("/") if segment]
    if "rag" in segments:
        return parse.urlunsplit(parsed)
    return parse.urlunsplit(parsed._replace(path="/" + "/".join([*segments, "rag"])))


def _encode_multipart(files: Sequence[UploadFile], fields: Mapping[str, str]) -> tuple[bytes, str]:
    boundary = f"----SeaRAG{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for file in files:
        if not file.name:
            raise ValueError("each upload file requires a name")
        content = file.content.read() if hasattr(file.content, "read") else file.content
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, bytes):
            raise TypeError("UploadFile.content must be bytes or a binary reader")
        escaped_name = file.name.replace('"', "%22")
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="file"; filename="{escaped_name}"\r\n'.encode("utf-8"),
                b"Content-Type: application/octet-stream\r\n\r\n",
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _raise_rag_response_error(value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    code = value.get("code")
    if isinstance(code, int) and code != 0:
        raise APIError(str(value.get("message") or "RAGFlow request failed"), code=code)


def _http_error(status_code: int, raw: str) -> APIError:
    try:
        response = json.loads(raw)
    except json.JSONDecodeError:
        response = {}
    if isinstance(response, Mapping):
        message = response.get("error") or response.get("message") or raw
    else:
        message = raw
    return APIError(str(message), status_code=status_code)


def _collapse_path(path: str) -> str:
    return "/" + "/".join(segment for segment in path.split("/") if segment)


def _query_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).lower() if isinstance(item, bool) else str(item) for item in value if not _is_zero_value(item)]
    return [str(value).lower() if isinstance(value, bool) else str(value)]


def _is_zero_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value == 0
    return False


def _has_header(headers: Mapping[str, str], name: str) -> bool:
    return any(key.lower() == name.lower() for key in headers)


def _is_debug_enabled() -> bool:
    return os.environ.get("SEARAG_DEBUG") == "1"


def _response_preview(raw: str) -> str:
    value = " ".join(raw.split())
    return value[:240]
