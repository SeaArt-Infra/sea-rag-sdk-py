from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, BinaryIO, Generic, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ClientOptions:
    """Client configuration. endpoint is a gateway base URL, not an API path."""

    endpoint: str
    api_key: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 180.0


@dataclass(frozen=True)
class RAGResponse(Generic[T]):
    """The normalized RAGFlow response envelope returned by resource methods."""

    code: int
    data: T
    message: str = ""
    total_datasets: int | None = None

    @property
    def success(self) -> bool:
        return self.code == 0


@dataclass(frozen=True)
class UploadFile:
    """One file to send in RAGFlow's repeated multipart form field ``file``."""

    name: str
    content: bytes | BinaryIO


@dataclass(frozen=True)
class UploadedFile:
    """An attachment created from a URL without creating a dataset document."""

    id: str = ""
    name: str = ""
    size: int = 0
    extension: str = ""
    mime_type: str = ""
    created_by: str = ""
    created_at: float = 0.0
    preview_url: str = ""


@dataclass(frozen=True)
class Dataset:
    id: str = ""
    name: str = ""
    description: str = ""
    embedding_model: str = ""
    permission: str = ""
    chunk_method: str = ""
    document_count: int = 0
    chunk_count: int = 0


@dataclass(frozen=True)
class Document:
    id: str = ""
    name: str = ""
    dataset_id: str = ""
    run: str = "UNSTART"
    progress: float = 0.0
    progress_msg: str = ""
    status: str = ""
    chunk_count: int = 0
    token_count: int = 0

    @property
    def parsing_status(self) -> str:
        return normalize_parsing_status(self.run)


@dataclass(frozen=True)
class DocumentList:
    total: int = 0
    docs: list[Document] | None = None

    def __post_init__(self) -> None:
        if self.docs is None:
            object.__setattr__(self, "docs", [])


@dataclass(frozen=True)
class Chunk:
    id: str = ""
    content: str = ""
    dataset_id: str = ""
    document_id: str = ""
    document_name: str = ""
    document_keyword: str = ""
    important_keywords: list[str] | None = None
    questions: list[str] | None = None
    tag_keywords: list[str] | None = None
    similarity: float = 0.0
    vector_similarity: float = 0.0
    term_similarity: float = 0.0
    available: bool = False

    def __post_init__(self) -> None:
        for name in ("important_keywords", "questions", "tag_keywords"):
            if getattr(self, name) is None:
                object.__setattr__(self, name, [])


@dataclass(frozen=True)
class ChunkList:
    total: int = 0
    chunks: list[Chunk] | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            object.__setattr__(self, "chunks", [])


@dataclass(frozen=True)
class DocumentAggregation:
    count: int = 0
    document_id: str = ""
    document_name: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    total: int = 0
    chunks: list[Chunk] | None = None
    document_aggregations: list[DocumentAggregation] | None = None

    def __post_init__(self) -> None:
        if self.chunks is None:
            object.__setattr__(self, "chunks", [])
        if self.document_aggregations is None:
            object.__setattr__(self, "document_aggregations", [])


@dataclass(frozen=True)
class Chat:
    id: str = ""
    name: str = ""
    dataset_ids: list[str] | None = None
    llm_id: str = ""

    def __post_init__(self) -> None:
        if self.dataset_ids is None:
            object.__setattr__(self, "dataset_ids", [])


@dataclass(frozen=True)
class ChatList:
    total: int = 0
    chats: list[Chat] | None = None

    def __post_init__(self) -> None:
        if self.chats is None:
            object.__setattr__(self, "chats", [])


@dataclass(frozen=True)
class DatasetListOptions:
    page: int | None = None
    page_size: int | None = None
    orderby: str | None = None
    desc: bool | None = None
    id: str | None = None
    name: str | None = None
    include_parsing_status: bool | None = None


@dataclass(frozen=True)
class DocumentListOptions:
    page: int | None = None
    page_size: int | None = None
    orderby: str | None = None
    desc: bool | None = None
    id: str | None = None
    ids: list[str] | None = None
    name: str | None = None
    keywords: str | None = None
    create_time_from: int | None = None
    create_time_to: int | None = None
    suffix: str | None = None
    run: str | None = None


@dataclass(frozen=True)
class ChunkListOptions:
    page: int | None = None
    page_size: int | None = None
    id: str | None = None
    keywords: str | None = None


@dataclass(frozen=True)
class ChatListOptions:
    page: int | None = None
    page_size: int | None = None
    orderby: str | None = None
    desc: bool | None = None
    id: str | None = None
    name: str | None = None
    keywords: str | None = None


@dataclass(frozen=True)
class WaitForParsingOptions:
    """Controls polling for one document's asynchronous RAGFlow parse job."""

    poll_interval: float = 1.0
    timeout: float = 15 * 60
    on_progress: Callable[[Document], None] | None = None


def normalize_parsing_status(value: object) -> str:
    """Normalize RAGFlow's textual and numeric parsing-status forms."""

    status = str(value or "").strip().upper()
    return {
        "0": "UNSTART",
        "1": "RUNNING",
        "2": "CANCEL",
        "3": "DONE",
        "4": "FAIL",
        "5": "SCHEDULE",
        "CANCELLED": "CANCEL",
        "FAILED": "FAIL",
        "SCHEDULED": "SCHEDULE",
    }.get(status, status)


def dataset_from_dict(value: Mapping[str, Any] | None) -> Dataset:
    raw = value or {}
    return Dataset(
        id=_text(raw.get("id")),
        name=_text(raw.get("name")),
        description=_text(raw.get("description")),
        embedding_model=_text(raw.get("embedding_model")),
        permission=_text(raw.get("permission")),
        chunk_method=_text(raw.get("chunk_method")),
        document_count=_integer(raw.get("document_count")),
        chunk_count=_integer(raw.get("chunk_count")),
    )


def document_from_dict(value: Mapping[str, Any] | None) -> Document:
    raw = value or {}
    return Document(
        id=_text(raw.get("id")),
        name=_text(raw.get("name")),
        dataset_id=_text(raw.get("dataset_id")),
        run=normalize_parsing_status(raw.get("run")) or "UNSTART",
        progress=_number(raw.get("progress")),
        progress_msg=_text(raw.get("progress_msg")),
        status=_text(raw.get("status")),
        chunk_count=_integer(raw.get("chunk_count")),
        token_count=_integer(raw.get("token_count")),
    )


def uploaded_file_from_dict(value: Mapping[str, Any] | None) -> UploadedFile:
    raw = value or {}
    return UploadedFile(
        id=_text(raw.get("id")),
        name=_text(raw.get("name")),
        size=_integer(raw.get("size")),
        extension=_text(raw.get("extension")),
        mime_type=_text(raw.get("mime_type")),
        created_by=_text(raw.get("created_by")),
        created_at=_number(raw.get("created_at")),
        preview_url=_text(raw.get("preview_url")),
    )


def document_list_from_dict(value: Mapping[str, Any] | None) -> DocumentList:
    raw = value or {}
    return DocumentList(
        total=_integer(raw.get("total")),
        docs=[document_from_dict(item) for item in _mappings(raw.get("docs"))],
    )


def chunk_from_dict(value: Mapping[str, Any] | None) -> Chunk:
    raw = value or {}
    return Chunk(
        id=_text(raw.get("id")),
        content=_text(raw.get("content")),
        dataset_id=_text(raw.get("dataset_id")),
        document_id=_text(raw.get("document_id")),
        document_name=_text(raw.get("document_name")),
        document_keyword=_text(raw.get("document_keyword")),
        important_keywords=_texts(raw.get("important_keywords")),
        questions=_texts(raw.get("questions")),
        tag_keywords=_texts(raw.get("tag_kwd")),
        similarity=_number(raw.get("similarity")),
        vector_similarity=_number(raw.get("vector_similarity")),
        term_similarity=_number(raw.get("term_similarity")),
        available=bool(raw.get("available", False)),
    )


def chunk_list_from_dict(value: Mapping[str, Any] | None) -> ChunkList:
    raw = value or {}
    return ChunkList(
        total=_integer(raw.get("total")),
        chunks=[chunk_from_dict(item) for item in _mappings(raw.get("chunks"))],
    )


def retrieval_result_from_dict(value: Mapping[str, Any] | None) -> RetrievalResult:
    raw = value or {}
    return RetrievalResult(
        total=_integer(raw.get("total")),
        chunks=[chunk_from_dict(item) for item in _mappings(raw.get("chunks"))],
        document_aggregations=[
            DocumentAggregation(
                count=_integer(item.get("count")),
                document_id=_text(item.get("doc_id")),
                document_name=_text(item.get("doc_name")),
            )
            for item in _mappings(raw.get("doc_aggs"))
        ],
    )


def chat_from_dict(value: Mapping[str, Any] | None) -> Chat:
    raw = value or {}
    return Chat(
        id=_text(raw.get("id")),
        name=_text(raw.get("name")),
        dataset_ids=_texts(raw.get("dataset_ids")),
        llm_id=_text(raw.get("llm_id")),
    )


def chat_list_from_dict(value: Mapping[str, Any] | None) -> ChatList:
    raw = value or {}
    return ChatList(
        total=_integer(raw.get("total")),
        chats=[chat_from_dict(item) for item in _mappings(raw.get("chats"))],
    )


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _texts(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [_text(item) for item in value]


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0
