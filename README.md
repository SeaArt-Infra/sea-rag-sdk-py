# sea-rag-sdk-py

Python SDK for SeaArt RAG. It provides RAGFlow API helpers for datasets, document ingestion, chunk curation, retrieval, and chat completion through the SeaArt gateway.

## Available Resources

| Client resource | Function |
| --- | --- |
| `client.datasets` | Create, list, get, update, and delete datasets |
| `client.documents` | Upload, list, update, parse, stop, delete, and download documents |
| `client.chunks` | Start or cancel parsing and manage chunks |
| `client.retrieval` | Retrieve grounded chunks from datasets |
| `client.chat` | Manage chat assistants and run JSON or streaming completions |
| `client.raw` | Call a RAGFlow endpoint not yet represented by a helper |

## Gateway Routing

Create the client with the SeaArt gateway base URL only. The SDK appends `/rag` once and each resource then appends `/api/v1`.

| Input endpoint | Request example |
| --- | --- |
| `https://gateway.example.com` | `https://gateway.example.com/rag/api/v1/retrieval` |
| `https://gateway.example.com/rag` | `https://gateway.example.com/rag/api/v1/retrieval` |
| `https://gateway.example.com/team-a` | `https://gateway.example.com/team-a/rag/api/v1/retrieval` |

The convention matches OpenResty's `/rag/` reverse-proxy route. Do not put `/api/v1` in `ClientOptions.endpoint`.

## Install

```bash
pip install sea-rag-sdk
```

Python 3.10 or newer is required. The runtime has no third-party dependency.

## Quick Start

```python
import os

import sea_rag_sdk as rag

client = rag.Client(
    rag.ClientOptions(
        endpoint=os.environ["SEAART_GATEWAY_BASE_URL"],
        api_key=os.environ["SEAART_RAG_API_KEY"],
        headers={"X-Project-ID": os.environ["RAGFLOW_PROJECT_ID"]},
    )
)

result = client.retrieval.search(
    {
        "dataset_ids": ["dataset-id"],
        "question": "What does the handbook say about leave?",
        "top_k": 20,
    }
)
print(result)
```

The SDK sends `Authorization: Bearer <api_key>` unless global `headers` already contains Authorization. This project-scoped RAGFlow deployment requires `X-Project-ID`; use `headers` for it and other gateway-wide trace or tenancy headers.

## Ingest A Document

RAGFlow expects each uploaded document in a repeated multipart form field named `file`. `UploadFile` accepts bytes or a binary file handle. Every JSON resource method returns `RAGResponse[T]`: the envelope keeps `code`, `message`, and `success`, while `data` is a typed resource payload.

```python
dataset_id = os.environ["RAGFLOW_DATASET_ID"]

with open("handbook.pdf", "rb") as source:
    uploaded = client.documents.upload(
        dataset_id,
        [rag.UploadFile("handbook.pdf", source)],
    )

if not uploaded.data:
    raise RuntimeError("RAGFlow upload returned no document")
document_id = uploaded.data[0].id
client.chunks.start_parsing(dataset_id, [document_id])
client.documents.wait_for_parsing(dataset_id, document_id)
```

Use `client.documents.parse()` and `client.documents.stop()` for newer document parse endpoints. Use `client.chunks.start_parsing()` and `client.chunks.cancel_parsing()` for RAGFlow-compatible chunk parse routes.

Parsing is asynchronous. Call `client.documents.wait_for_parsing()` before retrieval; it polls every second by default and waits for up to 15 minutes. It returns a typed `Document` on `DONE`, raises `sea_rag_sdk.ParsingFailedError` on `CANCEL` or `FAIL`, and raises `sea_rag_sdk.ParsingTimeoutError` on timeout. `WaitForParsingOptions.on_progress` receives every observed document state.

## Retrieve And Curate Chunks

Pass the RAGFlow retrieval body directly to `client.retrieval.search()`. Supported RAGFlow options include `dataset_ids`, `document_ids`, `question`, `similarity_threshold`, `vector_similarity_weight`, `top_k`, `rerank_id`, metadata conditions, and graph retrieval settings.

```python
result = client.retrieval.search(
    {
        "dataset_ids": ["dataset-id"],
        "question": "How are expenses approved?",
        "similarity_threshold": 0.2,
        "vector_similarity_weight": 0.3,
        "top_k": 20,
        "highlight": True,
    }
)
```

Use `client.chunks.list()`, `get()`, `create()`, `update()`, and `delete()` for curated chunks. Dataset, document, and chunk IDs are escaped by the SDK.

## Chat Completion And Streaming

Use `client.chat.complete()` for a non-streaming RAGFlow completion. `client.chat.stream()` adds `stream: True` and invokes the callback with UTF-8-safe raw SSE chunks. Keep event parsing in the application because RAGFlow event payloads vary with the chat configuration.

```python
client.chat.stream(
    {"chat_id": "chat-assistant-id", "question": "Summarize the leave policy."},
    lambda chunk: print(chunk, end=""),
)
```

## Errors And Unsupported APIs

`sea_rag_sdk.APIError` represents HTTP failures and RAGFlow responses whose `code` is non-zero. Inspect `status_code`, `code`, and `message` when handling errors. `wait_for_parsing()` additionally raises `ParsingFailedError` and `ParsingTimeoutError` for terminal parse outcomes.

For a non-core RAGFlow API, use the raw client with a full RAGFlow path:

```python
models = client.raw.request("GET", "/api/v1/models")
```

## Verify

```bash
python -m unittest discover -s tests -v
```

Do not place API keys in source control, browser code, logs, or telemetry. Avoid logging raw customer documents, prompts, and retrieved chunks.

<script
  type="text/plain"
  data-doc-skill
  data-doc-skill-id="sea-rag-sdk-py"
  data-doc-skill-label="SeaRAG Python SDK"
  data-doc-skill-filename="sea-rag-sdk-py-SKILL.md"
  data-doc-skill-version="1"
>
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
</script>
