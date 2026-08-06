from __future__ import annotations

from .resources import (
    ChatResource,
    ChunksResource,
    DatasetsResource,
    DocumentsResource,
    RawResource,
    RetrievalResource,
)
from .transport import Transport, normalize_rag_endpoint
from .types import ClientOptions


class Client:
    """RAGFlow API client routed through SeaArt's ``/rag`` gateway prefix."""

    def __init__(self, options: ClientOptions) -> None:
        self.endpoint = normalize_rag_endpoint(options.endpoint)
        self.api_key = options.api_key
        self.transport = Transport(
            self.endpoint,
            api_key=options.api_key,
            headers=options.headers,
            timeout=options.timeout,
        )
        self.datasets = DatasetsResource(self.transport)
        self.documents = DocumentsResource(self.transport)
        self.chunks = ChunksResource(self.transport)
        self.retrieval = RetrievalResource(self.transport)
        self.chat = ChatResource(self.transport)
        self.raw = RawResource(self.transport)


SeaRAGClient = Client
