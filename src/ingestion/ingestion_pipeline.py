# ============================================================
# src/ingestion/ingestion_pipeline.py - Document Ingestion Pipeline
#
# Learning Note:
#   INGESTION is the process of getting documents INTO the RAG system.
#   It runs OFFLINE (before queries) and involves:
#     1. Load document from file or GCS bucket
#     2. Extract and clean text
#     3. Split into chunks
#     4. Embed each chunk (convert to vector)
#     5. Store vectors + text + metadata in vector store
#     6. Update BM25 index
#
#   IMPORTANT: Ingestion only needs to run when you add new documents.
#   Retrieval (at query time) uses the pre-built indexes.
#   This offline/online separation is key to performance.
# ============================================================

import os
from pathlib import Path
from typing import List, Optional
from src.ingestion.document_loader import DocumentLoader
from src.observability.logger import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion: load -> chunk -> embed -> index.
    Supports local files and Google Cloud Storage.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.loader = DocumentLoader(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def ingest_file(self, file_path: str) -> dict:
        """
        Ingest a single local file into the RAG system.

        Returns:
            dict with ingestion stats (chunks_added, source, status)
        """
        logger.info("ingesting_file", path=file_path)

        # Step 1 & 2: Load and chunk
        chunks = self.loader.load_and_chunk(file_path)
        if not chunks:
            return {"status": "error", "message": f"Failed to load {file_path}", "chunks_added": 0}

        # Step 3 & 4: Embed and store in vector store
        from src.retrieval.vector_store import vector_store
        ids = vector_store.add_documents(chunks)

        # Step 5: Update BM25 index with all current documents
        # We fetch all docs for BM25 - in production use incremental update
        self._refresh_bm25_index()

        logger.info("file_ingested", path=file_path, chunks=len(chunks))
        return {
            "status": "success",
            "source": Path(file_path).name,
            "chunks_added": len(chunks),
            "document_ids": ids[:5],  # first 5 IDs for reference
        }

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> dict:
        """
        Register an uploaded document and retain its original for reindexing.
        """
        from src.sources.documents import document_service
        source = document_service.upload(file_bytes, filename)
        return {
            "status": "success",
            "source": source.filename,
            "source_id": source.id,
            "chunks_added": source.chunk_count,
        }

    def ingest_gcs_file(self, gcs_uri: str) -> dict:
        """
        Download a file from Google Cloud Storage and ingest it.
        gcs_uri format: gs://bucket-name/path/to/file.pdf

        Learning Note:
            GCS (Google Cloud Storage) is like S3 for GCP.
            In production, documents are stored in GCS buckets.
            A Pub/Sub trigger can call this automatically when
            a new file is uploaded to the bucket.
        """
        from src.config import settings
        from src import network_policy
        if not network_policy.external_allowed():
            return {"status": "error", "message": "Google Cloud Storage is outside the company network; "
                                                  "external calls are disabled (ALLOW_EXTERNAL_NETWORK=false)."}
        try:
            from google.cloud import storage

            # Parse GCS URI: gs://bucket/path
            if not gcs_uri.startswith("gs://"):
                return {"status": "error", "message": "Invalid GCS URI format"}

            parts = gcs_uri[5:].split("/", 1)
            bucket_name, blob_path = parts[0], parts[1]

            client = storage.Client(project=settings.gcp_project_id)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            # Download to temp file
            import tempfile
            filename = Path(blob_path).name
            suffix = Path(filename).suffix

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                blob.download_to_filename(tmp.name)
                tmp_path = tmp.name

            try:
                result = self.ingest_file(tmp_path)
                result["source"] = filename
                result["gcs_uri"] = gcs_uri
                return result
            finally:
                os.unlink(tmp_path)

        except ImportError:
            return {"status": "error", "message": "google-cloud-storage not installed"}
        except Exception as exc:
            logger.error("gcs_ingestion_failed", uri=gcs_uri, error=str(exc))
            return {"status": "error", "message": str(exc)}

    def ingest_directory(self, dir_path: str) -> dict:
        """Ingest all supported files from a local directory."""
        logger.info("ingesting_directory", path=dir_path)

        chunks = self.loader.load_directory(dir_path)
        if not chunks:
            return {"status": "error", "message": f"No documents found in {dir_path}", "chunks_added": 0}

        from src.retrieval.vector_store import vector_store
        ids = vector_store.add_documents(chunks)
        self._refresh_bm25_index()

        return {
            "status": "success",
            "directory": dir_path,
            "chunks_added": len(chunks),
        }

    def _refresh_bm25_index(self, strict: bool = False) -> None:
        """
        Rebuild BM25 index using documents in the vector store.
        Called after every ingestion to keep BM25 in sync.

        Learning Note:
            In production with large document sets, you'd maintain
            a separate document store (like Firestore) and rebuild
            BM25 incrementally. For learning, full rebuild is fine.
        """
        try:
            from src.retrieval.hybrid_retriever import hybrid_retriever
            from src.retrieval.vector_store import vector_store

            # Clear stale entries first, including when the last source was deleted.
            hybrid_retriever.update_bm25_index([])

            # Get all docs from Chroma for BM25 rebuild
            if hasattr(vector_store._store, "_collection"):
                collection = vector_store._store._collection
                data = collection.get(include=["documents", "metadatas"])

                from langchain.schema import Document
                all_docs = [
                    Document(page_content=text, metadata=meta or {})
                    for text, meta in zip(
                        data.get("documents", []),
                        data.get("metadatas", [{}] * len(data.get("documents", []))),
                    )
                ]

                hybrid_retriever.update_bm25_index(all_docs)
                logger.info("bm25_index_refreshed", doc_count=len(all_docs))

        except Exception as exc:
            logger.warning("bm25_refresh_failed", error=str(exc))
            if strict:
                raise

    def get_stats(self) -> dict:
        """Return current ingestion stats."""
        from src.retrieval.vector_store import vector_store
        return {
            "total_chunks": vector_store.get_document_count(),
            "vector_store_type": vector_store.collection_name,
        }


# Singleton instance
ingestion_pipeline = IngestionPipeline()
