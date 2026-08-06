from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, TypeVar

from .errors import ParsingFailedError, ParsingTimeoutError, RAGError
from .transport import QueryParams, Transport
from .types import (
    Chat,
    ChatList,
    ChatListOptions,
    Chunk,
    ChunkList,
    ChunkListOptions,
    Dataset,
    DatasetListOptions,
    Document,
    DocumentList,
    DocumentListOptions,
    RAGResponse,
    RetrievalResult,
    UploadFile,
    WaitForParsingOptions,
    chat_from_dict,
    chat_list_from_dict,
    chunk_from_dict,
    chunk_list_from_dict,
    dataset_from_dict,
    document_from_dict,
    document_list_from_dict,
    retrieval_result_from_dict,
)


API_PREFIX = "/api/v1"
DEFAULT_PARSE_POLL_INTERVAL = 1.0
DEFAULT_PARSE_WAIT_TIMEOUT = 15 * 60.0

T = TypeVar("T")


class _Resource:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: QueryParams | None = None,
        body: Any = None,
    ) -> Any:
        return self._transport.request_json(method, API_PREFIX + path, query=query, body=body)

    def _response(
        self,
        method: str,
        path: str,
        decode: Callable[[Any], T],
        *,
        query: QueryParams | None = None,
        body: Any = None,
    ) -> RAGResponse[T]:
        return _rag_response(self._request(method, path, query=query, body=body), decode)


class DatasetsResource(_Resource):
    def create(self, payload: Mapping[str, Any]) -> RAGResponse[Dataset]:
        return self._response("POST", "/datasets", dataset_from_dict, body=payload)

    def list(self, options: DatasetListOptions | Mapping[str, Any] | None = None) -> RAGResponse[list[Dataset]]:
        return self._response(
            "GET",
            "/datasets",
            lambda value: [dataset_from_dict(item) for item in _objects(value)],
            query=_options_query(options),
        )

    def get(self, dataset_id: str) -> RAGResponse[Dataset]:
        return self._response("GET", f"/datasets/{_path_segment(dataset_id)}", dataset_from_dict)

    def update(self, dataset_id: str, payload: Mapping[str, Any]) -> RAGResponse[Dataset]:
        return self._response("PUT", f"/datasets/{_path_segment(dataset_id)}", dataset_from_dict, body=payload)

    def delete(self, ids: Sequence[str] | None = None, *, delete_all: bool = False) -> RAGResponse[Any]:
        body: dict[str, Any] = {"delete_all": delete_all}
        if ids:
            body["ids"] = list(ids)
        return self._response("DELETE", "/datasets", _identity, body=body)


class DocumentsResource(_Resource):
    def upload(self, dataset_id: str, files: Sequence[UploadFile]) -> RAGResponse[list[Document]]:
        raw = self._transport.post_multipart(
            f"{API_PREFIX}/datasets/{_path_segment(dataset_id)}/documents", files
        )
        return _rag_response(raw, lambda value: [document_from_dict(item) for item in _objects(value)])

    def list(
        self,
        dataset_id: str,
        options: DocumentListOptions | Mapping[str, Any] | None = None,
    ) -> RAGResponse[DocumentList]:
        return self._response(
            "GET",
            f"/datasets/{_path_segment(dataset_id)}/documents",
            document_list_from_dict,
            query=_options_query(options),
        )

    def update(
        self,
        dataset_id: str,
        document_id: str,
        payload: Mapping[str, Any],
    ) -> RAGResponse[Document]:
        return self._response(
            "PATCH",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}",
            document_from_dict,
            body=payload,
        )

    def delete(
        self,
        dataset_id: str,
        ids: Sequence[str] | None = None,
        *,
        delete_all: bool = False,
    ) -> RAGResponse[Any]:
        body: dict[str, Any] = {"delete_all": delete_all}
        if ids:
            body["ids"] = list(ids)
        return self._response("DELETE", f"/datasets/{_path_segment(dataset_id)}/documents", _identity, body=body)

    def parse(self, dataset_id: str, document_ids: Sequence[str]) -> RAGResponse[Any]:
        return self._response(
            "POST",
            f"/datasets/{_path_segment(dataset_id)}/documents/parse",
            _identity,
            body={"document_ids": list(document_ids)},
        )

    def stop(self, dataset_id: str, document_ids: Sequence[str]) -> RAGResponse[Any]:
        return self._response(
            "POST",
            f"/datasets/{_path_segment(dataset_id)}/documents/stop",
            _identity,
            body={"document_ids": list(document_ids)},
        )

    def download(self, dataset_id: str, document_id: str) -> tuple[bytes, Mapping[str, str]]:
        return self._transport.download(
            f"{API_PREFIX}/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}"
        )

    def wait_for_parsing(
        self,
        dataset_id: str,
        document_id: str,
        options: WaitForParsingOptions | None = None,
    ) -> Document:
        """Wait until one document reaches DONE, or raise for a terminal failure."""

        if not isinstance(dataset_id, str) or not dataset_id.strip() or not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("dataset_id and document_id are required")
        options = options or WaitForParsingOptions()
        if not isinstance(options, WaitForParsingOptions):
            raise TypeError("options must be WaitForParsingOptions or None")
        interval = _positive_seconds(options.poll_interval, DEFAULT_PARSE_POLL_INTERVAL, "poll_interval")
        timeout = _positive_seconds(options.timeout, DEFAULT_PARSE_WAIT_TIMEOUT, "timeout")
        deadline = time.monotonic() + timeout
        last_document: Document | None = None

        while True:
            response = self.list(dataset_id, DocumentListOptions(id=document_id, page=1, page_size=1))
            if response.data.docs:
                document = response.data.docs[0]
                last_document = document
                if options.on_progress is not None:
                    options.on_progress(document)
                status = document.parsing_status
                if status == "DONE":
                    return document
                if status in {"CANCEL", "FAIL"}:
                    raise ParsingFailedError(document)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ParsingTimeoutError(dataset_id, document_id, timeout, last_document)
            time.sleep(min(interval, remaining))


class ChunksResource(_Resource):
    def start_parsing(self, dataset_id: str, document_ids: Sequence[str]) -> RAGResponse[Any]:
        return self._response(
            "POST",
            f"/datasets/{_path_segment(dataset_id)}/chunks",
            _identity,
            body={"document_ids": list(document_ids)},
        )

    def cancel_parsing(self, dataset_id: str, document_ids: Sequence[str]) -> RAGResponse[Any]:
        return self._response(
            "DELETE",
            f"/datasets/{_path_segment(dataset_id)}/chunks",
            _identity,
            body={"document_ids": list(document_ids)},
        )

    def list(
        self,
        dataset_id: str,
        document_id: str,
        options: ChunkListOptions | Mapping[str, Any] | None = None,
    ) -> RAGResponse[ChunkList]:
        return self._response(
            "GET",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}/chunks",
            chunk_list_from_dict,
            query=_options_query(options),
        )

    def get(self, dataset_id: str, document_id: str, chunk_id: str) -> RAGResponse[Chunk]:
        return self._response(
            "GET",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}/chunks/{_path_segment(chunk_id)}",
            chunk_from_dict,
        )

    def create(self, dataset_id: str, document_id: str, payload: Mapping[str, Any]) -> RAGResponse[Any]:
        return self._response(
            "POST",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}/chunks",
            _identity,
            body=payload,
        )

    def update(self, dataset_id: str, document_id: str, chunk_id: str, payload: Mapping[str, Any]) -> RAGResponse[Any]:
        return self._response(
            "PATCH",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}/chunks/{_path_segment(chunk_id)}",
            _identity,
            body=payload,
        )

    def delete(
        self,
        dataset_id: str,
        document_id: str,
        chunk_ids: Sequence[str] | None = None,
        *,
        delete_all: bool = False,
    ) -> RAGResponse[Any]:
        body: dict[str, Any] = {"delete_all": delete_all}
        if chunk_ids:
            body["chunk_ids"] = list(chunk_ids)
        return self._response(
            "DELETE",
            f"/datasets/{_path_segment(dataset_id)}/documents/{_path_segment(document_id)}/chunks",
            _identity,
            body=body,
        )


class RetrievalResource(_Resource):
    def search(self, payload: Mapping[str, Any]) -> RAGResponse[RetrievalResult]:
        return self._response("POST", "/retrieval", retrieval_result_from_dict, body=payload)


class ChatResource(_Resource):
    def create(self, payload: Mapping[str, Any]) -> RAGResponse[Chat]:
        return self._response("POST", "/chats", chat_from_dict, body=payload)

    def list(self, options: ChatListOptions | Mapping[str, Any] | None = None) -> RAGResponse[ChatList]:
        return self._response("GET", "/chats", chat_list_from_dict, query=_options_query(options))

    def get(self, chat_id: str) -> RAGResponse[Chat]:
        return self._response("GET", f"/chats/{_path_segment(chat_id)}", chat_from_dict)

    def update(self, chat_id: str, payload: Mapping[str, Any]) -> RAGResponse[Any]:
        return self._response("PATCH", f"/chats/{_path_segment(chat_id)}", _identity, body=payload)

    def delete(self, ids: Sequence[str] | None = None, *, delete_all: bool = False) -> RAGResponse[Any]:
        body: dict[str, Any] = {"delete_all": delete_all}
        if ids:
            body["ids"] = list(ids)
        return self._response("DELETE", "/chats", _identity, body=body)

    def complete(self, payload: Mapping[str, Any]) -> RAGResponse[Any]:
        return self._response("POST", "/chat/completions", _identity, body=payload)

    def stream(self, payload: Mapping[str, Any], on_chunk: Callable[[str], None]) -> None:
        body = dict(payload)
        body["stream"] = True
        self._transport.post_stream(f"{API_PREFIX}/chat/completions", body, on_chunk)


class RawResource:
    """Access an RAGFlow route which has no first-class SDK helper yet."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    def request(
        self,
        method: str,
        path: str,
        *,
        query: QueryParams | None = None,
        body: Any = None,
    ) -> RAGResponse[Any]:
        return _rag_response(self._transport.request_json(method, path, query=query, body=body), _identity)


def _rag_response(value: Any, decode: Callable[[Any], T]) -> RAGResponse[T]:
    if not isinstance(value, Mapping):
        raise RAGError("expected a RAGFlow response envelope")
    code = value.get("code", 0)
    if not isinstance(code, int):
        try:
            code = int(code)
        except (TypeError, ValueError) as exc:
            raise RAGError("RAGFlow response code must be an integer") from exc
    total_datasets = _optional_integer(value.get("total_datasets"))
    return RAGResponse(
        code=code,
        data=decode(value.get("data")),
        message=str(value.get("message") or ""),
        total_datasets=total_datasets,
    )


def _identity(value: T) -> T:
    return value


def _objects(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _options_query(options: Any) -> QueryParams | None:
    if options is None:
        return None
    if is_dataclass(options):
        return asdict(options)
    if isinstance(options, Mapping):
        return options
    raise TypeError("options must be a supported options dataclass or mapping")


def _path_segment(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _positive_seconds(value: float, default: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        return default
    return value


def _optional_integer(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
