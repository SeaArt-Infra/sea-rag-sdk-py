---
name: sea-rag-sdk-py
description: Integrate Python services with SeaArt RAG and RAGFlow through the official sea-rag-sdk package. Use for dataset creation, document upload and parsing, parsing-status waits, chunk management, retrieval, chat completion, or RAGFlow API access in Python 3.10+.
---

# SeaRAG Python SDK

Use `sea-rag-sdk` and `sea_rag_sdk` instead of hand-written RAGFlow HTTP calls.

## Workflow

1. Add the package with `pip install sea-rag-sdk`.
2. Create one `rag.Client` with the SeaArt gateway base URL and API key.
3. Pass the gateway base URL only. The SDK appends `/rag` once and resource methods add `/api/v1`.
4. Resource methods return typed `rag.RAGResponse[T]`; read payloads from `response.data`.
5. Call `client.documents.wait_for_parsing()` after starting parsing and before retrieval.
6. Run `python -m unittest discover -s tests -v` after changing the integration.

The client adds `Authorization: Bearer <api_key>` unless global headers supply Authorization. This project-scoped RAGFlow deployment also requires `X-Project-ID`, supplied through `headers`. Do not include `/rag` or `/api/v1` in normal endpoint configuration.

## Shortest Runnable Flow

Set `SEAART_GATEWAY_BASE_URL`, `SEAART_RAG_API_KEY`, `RAGFLOW_PROJECT_ID`, and `RAGFLOW_DATASET_ID`, then run this beside `handbook.pdf`:

```python
import os

import sea_rag_sdk as rag


dataset_id = os.environ["RAGFLOW_DATASET_ID"]
client = rag.Client(
    rag.ClientOptions(
        endpoint=os.environ["SEAART_GATEWAY_BASE_URL"],
        api_key=os.environ["SEAART_RAG_API_KEY"],
        headers={"X-Project-ID": os.environ["RAGFLOW_PROJECT_ID"]},
    )
)

with open("handbook.pdf", "rb") as source:
    uploaded = client.documents.upload(
        dataset_id,
        [rag.UploadFile("handbook.pdf", source)],
    )
if not uploaded.data:
    raise RuntimeError("RAGFlow upload returned no document")

document_id = uploaded.data[0].id
client.chunks.start_parsing(dataset_id, [document_id])
client.documents.wait_for_parsing(
    dataset_id,
    document_id,
    rag.WaitForParsingOptions(
        on_progress=lambda document: print(
            document.parsing_status, f"{document.progress:.0%}"
        ),
    ),
)

result = client.retrieval.search(
    {
        "dataset_ids": [dataset_id],
        "document_ids": [document_id],
        "question": "What does the handbook say about leave?",
        "top_k": 5,
    }
)
print(f"retrieved {len(result.data.chunks)} chunks")
```

`wait_for_parsing()` polls every second by default for up to 15 minutes. It returns a typed `Document` on `DONE`, invokes `on_progress` for every observed document state, raises `rag.ParsingFailedError` for `CANCEL` or `FAIL`, and raises `rag.ParsingTimeoutError` on timeout. RAGFlow state is normalized to `UNSTART`, `RUNNING`, `CANCEL`, `DONE`, or `FAIL`.

## Other Resources

Use `client.documents.parse()` and `stop()` for newer document parse endpoints. Use `client.chunks.cancel_parsing()` to cancel compatible parsing and its chunk CRUD helpers for curation. Use `client.chat.complete()` or `stream()` for configured chat assistants, and `client.raw.request(method, "/api/v1/...", query=..., body=...)` for an uncovered RAGFlow API.

## Safety

`rag.APIError` represents HTTP failures and RAGFlow envelopes whose `code` is non-zero. Keep API keys, customer documents, prompts, and raw retrieval output out of source control and logs.
