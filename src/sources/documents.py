"""Registered document uploads with retained originals for reindexing."""

from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from src.ingestion.document_loader import DocumentLoader
from src.sources.models import DocumentSourceConfig, DocumentSourceResponse, SourceStatus, SourceType
from src.sources.registry import source_registry
from src.observability.logger import get_logger

logger = get_logger(__name__)
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_FILE_BYTES = 50 * 1024 * 1024


class DocumentSourceService:
    def __init__(self, registry=None, storage_dir=None, vector_store=None, refresh=None):
        self.registry = registry if registry is not None else source_registry
        self.storage_dir = Path(storage_dir or "data/documents")
        self.loader = DocumentLoader()
        self._vector_store = vector_store
        self._refresh = refresh
        # Serializes reindex/delete within the local, single-worker application.
        self._lock = RLock()

    @property
    def vectors(self):
        if self._vector_store is None:
            from src.retrieval.vector_store import vector_store
            self._vector_store = vector_store
        return self._vector_store

    def refresh_search(self):
        if self._refresh is not None:
            self._refresh()
        else:
            from src.ingestion.ingestion_pipeline import ingestion_pipeline
            ingestion_pipeline._refresh_bm25_index(strict=True)

    def get(self, source_id):
        source = self.registry.get_source(source_id)
        config = self.registry.get_document_config(source_id)
        if not source or source.type != SourceType.DOCUMENT or not config:
            raise KeyError("Document source not found")
        return DocumentSourceResponse(
            **source.model_dump(), filename=config.filename,
            file_type=config.file_type, size_bytes=config.size_bytes,
        )

    def _path(self, source):
        # Paths use generated IDs, never user-supplied filenames.
        path = self.storage_dir / f"{source.id}.{source.file_type}"
        if path.resolve().parent != self.storage_dir.resolve():
            raise ValueError("Invalid document storage path")
        return path

    def upload(self, content: bytes, filename: str):
        filename = Path((filename or "").replace("\\", "/")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, or MD.")
        if not content:
            raise ValueError("The uploaded file is empty")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 50 MB limit")
        source_id = f"doc-{uuid4().hex}"
        with self._lock:
            self.registry.create_document_source(
                DocumentSourceConfig(source_id=source_id, filename=filename,
                                     file_type=suffix[1:], size_bytes=len(content)),
                name=filename,
            )
            try:
                self.storage_dir.mkdir(parents=True, exist_ok=True)
                path = self._path(self.get(source_id))
                with path.open("xb") as target:
                    target.write(content)
                return self.reindex(source_id)
            except Exception:
                self.registry.update_source_status(
                    source_id, SourceStatus.ERROR,
                    error_message="Upload or indexing failed. Check the file and retry reindexing.",
                )
                raise

    def reindex(self, source_id):
        with self._lock:
            source = self.get(source_id)
            path = self._path(source)
            if not path.is_file():
                self.registry.update_source_status(
                    source_id, SourceStatus.ERROR,
                    error_message="Original file is missing. Upload the document again.",
                )
                raise FileNotFoundError("Original document is missing; upload the file again")
            self.registry.update_source_status(source_id, SourceStatus.INDEXING)
            try:
                documents = self.loader.load_file(str(path))
                chunks = self.loader.chunk_documents(documents, source.filename)
                if not chunks:
                    raise ValueError("No readable text found. Check the file; scanned PDFs need OCR.")
                for chunk in chunks:
                    chunk.metadata.update(
                        chunk_id=chunk.metadata["document_id"],
                        source_id=source_id, source_type="document", source_name=source.name,
                        source_file=source.filename, filename=source.filename,
                        document_id=source_id, file_type=source.file_type,
                        page_number=chunk.metadata.get("page", 0) + 1,
                    )
                self.vectors.replace_source_documents(source_id, chunks)
                self.refresh_search()
                self.registry.update_source_status(
                    source_id, SourceStatus.READY, chunk_count=len(chunks),
                    last_sync=datetime.now(timezone.utc),
                )
                return self.get(source_id)
            except Exception:
                self.registry.update_source_status(
                    source_id, SourceStatus.ERROR,
                    error_message="Indexing failed. Check the original file and retry.",
                )
                logger.exception("document_index_failed", source_id=source_id)
                raise

    def delete(self, source_id):
        with self._lock:
            source = self.get(source_id)
            # Do not lose the registry or original if vector deletion fails.
            self.vectors.delete_by_source_id(source_id)
            self.refresh_search()
            self._path(source).unlink(missing_ok=True)
            self.registry.delete_source(source_id)


document_service = DocumentSourceService()
