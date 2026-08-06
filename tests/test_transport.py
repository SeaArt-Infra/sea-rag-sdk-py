from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import sea_rag_sdk as rag


class TransportTests(unittest.TestCase):
    def test_normalize_rag_endpoint(self) -> None:
        cases = [
            ("http://127.0.0.1:8080", "http://127.0.0.1:8080/rag"),
            ("http://127.0.0.1:8080/", "http://127.0.0.1:8080/rag"),
            ("http://127.0.0.1:8080/rag", "http://127.0.0.1:8080/rag"),
            ("http://127.0.0.1:8080/rag/", "http://127.0.0.1:8080/rag/"),
            ("https://example.com/api?debug=1", "https://example.com/api/rag?debug=1"),
            ("", ""),
        ]
        for endpoint, expected in cases:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(rag.normalize_rag_endpoint(endpoint), expected)

    def test_build_url_and_headers(self) -> None:
        transport = rag.Transport(
            "https://example.com/base?debug=1",
            api_key="secret",
            headers={"X-User-ID": "user_1"},
        )
        self.assertEqual(
            transport.build_url(
                "/api/v1/datasets/a%2Fb",
                {"ids": ["one", "two"], "desc": False, "page": 0},
            ),
            "https://example.com/base/rag/api/v1/datasets/a%2Fb?debug=1&ids=one&ids=two&desc=false",
        )
        headers = transport.build_headers("application/json", True, {"authorization": "Bearer custom"})
        self.assertEqual(headers["X-User-ID"], "user_1")
        self.assertEqual(headers["authorization"], "Bearer custom")
        self.assertNotIn("Authorization", headers)

    def test_upload_uses_gateway_route_multipart_and_bearer_auth(self) -> None:
        response = _Response(b'{"code": 0, "data": [{"id": "doc_1", "run": "1"}]}')
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example", api_key="api-key"))
        with patch("sea_rag_sdk.transport.request.urlopen", return_value=response) as urlopen:
            result = client.documents.upload("kb_1", [rag.UploadFile("notes.txt", b"rag content")])

        self.assertIsInstance(result, rag.RAGResponse)
        self.assertTrue(result.success)
        self.assertIsInstance(result.data[0], rag.Document)
        self.assertEqual(result.data[0].parsing_status, "RUNNING")
        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "https://gateway.example/rag/api/v1/datasets/kb_1/documents")
        self.assertEqual(req.get_header("Authorization"), "Bearer api-key")
        self.assertIn("multipart/form-data", req.get_header("Content-type"))
        self.assertIn(b'filename="notes.txt"', req.data)
        self.assertIn(b"rag content", req.data)

    def test_nonzero_rag_code_raises_api_error(self) -> None:
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example"))
        with patch(
            "sea_rag_sdk.transport.request.urlopen",
            return_value=_Response(b'{"code": 102, "message": "dataset_ids is required"}'),
        ):
            with self.assertRaises(rag.APIError) as raised:
                client.retrieval.search({})
        self.assertEqual(raised.exception.code, 102)

    def test_chat_stream_forces_stream_and_handles_split_utf8(self) -> None:
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example"))
        response = _ChunkedResponse([b"data: \xe4", b"\xbd\xa0\n\n", b""])
        chunks: list[str] = []
        with patch("sea_rag_sdk.transport.request.urlopen", return_value=response) as urlopen:
            client.chat.stream({"chat_id": "chat_1", "question": "hello"}, chunks.append)

        req = urlopen.call_args.args[0]
        self.assertEqual(req.full_url, "https://gateway.example/rag/api/v1/chat/completions")
        self.assertTrue(json.loads(req.data)["stream"])
        self.assertEqual("".join(chunks), "data: 你\n\n")

    def test_wait_for_parsing_returns_typed_done_document(self) -> None:
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example"))
        progress: list[str] = []
        running = rag.RAGResponse(
            code=0,
            data=rag.DocumentList(docs=[rag.Document(id="doc_1", run="RUNNING")]),
        )
        done = rag.RAGResponse(
            code=0,
            data=rag.DocumentList(docs=[rag.Document(id="doc_1", run="DONE", chunk_count=3)]),
        )
        with patch.object(client.documents, "list", side_effect=[running, done]) as listed:
            document = client.documents.wait_for_parsing(
                "kb_1",
                "doc_1",
                rag.WaitForParsingOptions(
                    poll_interval=0.001,
                    timeout=1,
                    on_progress=lambda item: progress.append(item.parsing_status),
                ),
            )

        self.assertEqual(document.parsing_status, "DONE")
        self.assertEqual(document.chunk_count, 3)
        self.assertEqual(progress, ["RUNNING", "DONE"])
        first_call = listed.call_args_list[0].args
        self.assertEqual(first_call[0], "kb_1")
        self.assertEqual(first_call[1].id, "doc_1")

    def test_wait_for_parsing_raises_on_failed_document(self) -> None:
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example"))
        failed = rag.RAGResponse(
            code=0,
            data=rag.DocumentList(
                docs=[rag.Document(id="doc_1", run="4", progress_msg="invalid file")]
            ),
        )
        with patch.object(client.documents, "list", return_value=failed):
            with self.assertRaises(rag.ParsingFailedError) as raised:
                client.documents.wait_for_parsing("kb_1", "doc_1")

        self.assertEqual(raised.exception.document.parsing_status, "FAIL")

    def test_wait_for_parsing_raises_timeout_with_last_document(self) -> None:
        client = rag.Client(rag.ClientOptions(endpoint="https://gateway.example"))
        running = rag.RAGResponse(
            code=0,
            data=rag.DocumentList(docs=[rag.Document(id="doc_1", run="RUNNING")]),
        )
        with patch.object(client.documents, "list", return_value=running):
            with self.assertRaises(rag.ParsingTimeoutError) as raised:
                client.documents.wait_for_parsing(
                    "kb_1",
                    "doc_1",
                    rag.WaitForParsingOptions(poll_interval=0.001, timeout=0.003),
                )

        self.assertEqual(raised.exception.last_document.parsing_status, "RUNNING")


class _Response:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.headers: dict[str, str] = {}

    def read(self, size: int = -1) -> bytes:
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None


class _ChunkedResponse(_Response):
    def __init__(self, chunks: list[bytes]) -> None:
        super().__init__(b"")
        self.chunks = iter(chunks)

    def read(self, size: int = -1) -> bytes:
        return next(self.chunks)


if __name__ == "__main__":
    unittest.main()
