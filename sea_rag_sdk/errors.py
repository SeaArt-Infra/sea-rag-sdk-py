class RAGError(Exception):
    """Base exception for SeaRAG SDK failures."""


class APIError(RAGError):
    """An HTTP failure or a non-zero RAGFlow response code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: int | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        if code is not None:
            prefix = f"RAG API error {code}"
        elif status_code is not None:
            prefix = f"HTTP {status_code}"
        else:
            prefix = "RAG API error"
        super().__init__(f"{prefix}: {message}")


class ParsingFailedError(RAGError):
    """RAGFlow completed a parse job in CANCEL or FAIL state."""

    def __init__(self, document: object) -> None:
        self.document = document
        document_id = getattr(document, "id", "")
        state = getattr(document, "parsing_status", getattr(document, "run", ""))
        message = getattr(document, "progress_msg", "") or "RAGFlow parsing did not complete"
        super().__init__(f'document "{document_id}" parsing ended as {state}: {message}')


class ParsingTimeoutError(RAGError):
    """RAGFlow did not report a terminal parsing state before the timeout."""

    def __init__(self, dataset_id: str, document_id: str, timeout: float, last_document: object | None = None) -> None:
        self.dataset_id = dataset_id
        self.document_id = document_id
        self.timeout = timeout
        self.last_document = last_document
        super().__init__(f'document "{document_id}" was not parsed within {timeout:g} seconds')
