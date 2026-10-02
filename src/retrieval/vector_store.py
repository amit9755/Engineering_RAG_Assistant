# ============================================================
# src/retrieval/vector_store.py - Vector Store Abstraction
#
# Learning Note:
#   A vector store is a database optimised for similarity search.
#   When you search "what is machine learning?", it finds chunks
#   whose embedding vectors are closest to the query vector.
#
#   ChromaDB (default, FREE):
#     - Runs locally, no setup needed
#     - Persists data to disk
#     - Perfect for dev and learning
#
#   Vertex AI Vector Search (production, paid):
#     - Managed by Google, auto-scales
#     - Handles billions of vectors
#     - Used when you deploy to GCP
#
#   We use an abstraction layer so the rest of the code does not
#   care which backend is used - just call .search() and get results.
# ============================================================

from typing import List, Tuple
from langchain_core.documents import Document
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Abstraction over ChromaDB (local/dev) and Vertex AI Vector Search (production).
    Provides: add_documents(), similarity_search(), delete_collection()
    """

    ADD_BATCH_SIZE = 256

    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self._store = None
        self._embeddings = None
        self._init_store()

    def _init_store(self) -> None:
        """Initialise the underlying vector store based on config."""
        # Import embedder here to avoid circular imports at module load
        from src.ingestion.embedder import embedding_model
        self._embeddings = embedding_model.langchain_model

        if settings.vector_store_type == "chroma":
            self._init_chroma()
        elif settings.vector_store_type == "vertexai":
            self._init_vertexai()
        else:
            raise ValueError(f"Unknown vector_store_type: {settings.vector_store_type}")

    def _init_chroma(self) -> None:
        """
        Initialise ChromaDB with persistent storage.

        Learning Note:
            persist_directory means data survives app restarts.
            Without it, you'd have to re-ingest documents every run.
        """
        import os
        from langchain_community.vectorstores import Chroma

        os.makedirs(settings.chroma_persist_dir, exist_ok=True)

        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
        )
        logger.info(
            "chroma_initialized",
            collection=self.collection_name,
            persist_dir=settings.chroma_persist_dir,
        )

    def _init_vertexai(self) -> None:
        """
        Initialise Vertex AI Vector Search (production).

        Learning Note:
            Vertex AI Vector Search is a managed service - Google handles
            sharding, replication, and scaling. You only need to provide
            an Index ID and Endpoint ID from your GCP project.
        """
        try:
            from langchain_google_vertexai.vectorstores import VectorSearchVectorStore
            self._store = VectorSearchVectorStore.from_components(
                project_id=settings.gcp_project_id,
                region=settings.gcp_region,
                gcs_bucket_name=settings.gcp_bucket_name,
                index_id=settings.vertex_ai_index_id,
                endpoint_id=settings.vertex_ai_index_endpoint_id,
                embedding=self._embeddings,
            )
            logger.info("vertexai_vector_store_initialized", project=settings.gcp_project_id)
        except Exception as exc:
            logger.error("vertexai_init_failed", error=str(exc))
            raise

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add document chunks to the vector store.
        Returns list of IDs assigned to the added documents.

        Learning Note:
            Under the hood, LangChain calls embed_documents() on the text
            of each chunk and stores (vector, metadata, text) in the DB.
        """
        if not documents:
            logger.warning("add_documents_called_with_empty_list")
            return []

        logger.info("adding_documents", count=len(documents))
        ids = self._store.add_documents(documents)
        logger.info("documents_added", count=len(ids))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = None,
    ) -> List[Document]:
        """
        Search for the top-k most similar chunks to the query.

        Learning Note:
            This performs approximate nearest neighbor (ANN) search.
            The query is embedded first, then we find the k vectors
            in the store with the highest cosine similarity.
            k defaults to retrieval_top_k from config.
        """
        k = k or settings.retrieval_top_k
        results = self._store.similarity_search(query=query, k=k)
        logger.info("vector_search_complete", query_len=len(query), results=len(results))
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search and return (Document, similarity_score) pairs.
        Scores are between 0-1, higher = more similar.
        Used when we need scores for hybrid ranking.
        """
        k = k or settings.retrieval_top_k
        results = self._store.similarity_search_with_score(query=query, k=k)
        return results

    def delete_by_source_id(self, source_id: str) -> int:
        """
        Delete all chunks whose metadata contains the given source_id.
        Returns the number of chunks deleted.

        This is called when a source is removed from the registry so
        its vectors are also purged from ChromaDB.
        """
        if settings.vector_store_type != "chroma":
            raise NotImplementedError("Source deletion requires the Chroma backend")
        collection = self._store._collection
        ids = collection.get(where={"source_id": source_id}, include=[])["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)

    def replace_source_documents(self, source_id: str, documents: List[Document],
                                 on_progress=None) -> List[str]:
        """
        Index a new revision before removing old chunks; preserve old data on failure.
        on_progress(done, total) is called after each embedded batch.
        """
        import uuid

        if settings.vector_store_type != "chroma":
            raise NotImplementedError("Managed document indexing requires the Chroma backend")
        if not documents:
            raise ValueError("Document contains no extractable text")
        if any(doc.metadata.get("source_id") != source_id for doc in documents):
            raise ValueError("All replacement chunks must belong to the same source")
        collection = self._store._collection
        old_ids = collection.get(where={"source_id": source_id}, include=[])["ids"]
        new_ids = [str(uuid.uuid4()) for _ in documents]
        try:
            # Chroma rejects oversized batches, and repositories can produce thousands of chunks.
            for start in range(0, len(documents), self.ADD_BATCH_SIZE):
                end = start + self.ADD_BATCH_SIZE
                self._store.add_documents(documents[start:end], ids=new_ids[start:end])
                if on_progress:
                    on_progress(min(end, len(documents)), len(documents))
            if old_ids:
                collection.delete(ids=old_ids)
        except Exception:
            # Also handles a partial batched insertion before an embedding/store error.
            collection.delete(ids=new_ids)
            raise
        return new_ids

    def similarity_search_with_filter(
        self,
        query: str,
        source_filter,
        k: int = None,
    ) -> List[Document]:
        """
        Search for similar chunks restricted to the selected sources.

        This ensures queries NEVER return results from sources that
        were not selected by the user. A filter that fails is an error,
        never a silent fallback to searching everything.

        Args:
            query: search query text
            source_filter: SourceFilter, or None to search everything
            k: number of results (defaults to retrieval_top_k)
        """
        k = k or settings.retrieval_top_k

        if source_filter is None:
            return self.similarity_search(query, k=k)
        if source_filter.is_empty:
            return []

        if settings.vector_store_type == "chroma":
            results = self._store.similarity_search(
                query=query, k=k, filter=source_filter.chroma_where(),
            )
            logger.info("filtered_vector_search", results=len(results))
            return results
        # Fallback for Vertex AI - filter in Python after retrieval
        all_results = self.similarity_search(query, k=k * 3)
        return [doc for doc in all_results if source_filter.matches(doc.metadata)][:k]

    def get_commit_history(self, source_filter=None, chunks_per_source: int = 2) -> List[Document]:
        """Newest commit-history chunks of each (selected) repository, newest first."""
        if settings.vector_store_type != "chroma":
            return []
        where = {"file_path": "(commit history)"}
        if source_filter is not None:
            if not source_filter.source_ids:
                return []
            where = {"$and": [where, {"source_id": {"$in": source_filter.source_ids}}]}
        data = self._store._collection.get(where=where, include=["documents", "metadatas"])
        docs = [Document(page_content=text, metadata=meta)
                for text, meta in zip(data["documents"], data["metadatas"])
                if (meta or {}).get("chunk_index", 0) < chunks_per_source]
        return sorted(docs, key=lambda d: (d.metadata.get("source_id", ""), d.metadata.get("chunk_index", 0)))

    # ----------------------------------------------------------
    # Legacy chunks: uploaded before the source registry existed,
    # so they have no source_id. Grouped by their source_file.
    # ----------------------------------------------------------

    def list_legacy_files(self) -> List[dict]:
        """Group unregistered chunks by file, with a short preview of the first chunk."""
        if settings.vector_store_type != "chroma":
            return []
        data = self._store._collection.get(include=["metadatas", "documents"])
        groups = {}
        for text, meta in zip(data["documents"], data["metadatas"]):
            meta = meta or {}
            if meta.get("source_id"):
                continue
            name = meta.get("source_file") or meta.get("source") or "unknown"
            group = groups.setdefault(name, {"source_file": name, "chunk_count": 0,
                                             "preview": "", "_first": None})
            group["chunk_count"] += 1
            index = meta.get("chunk_index", 0)
            if group["_first"] is None or index < group["_first"]:
                group["_first"] = index
                group["preview"] = " ".join((text or "").split())[:160]
        for group in groups.values():
            del group["_first"]
        return sorted(groups.values(), key=lambda g: g["source_file"])

    def delete_legacy_file(self, source_file: str) -> int:
        """Delete unregistered chunks for one file; registered sources are never touched."""
        if settings.vector_store_type != "chroma":
            raise NotImplementedError("Legacy chunk deletion requires the Chroma backend")
        collection = self._store._collection
        data = collection.get(where={"source_file": source_file}, include=["metadatas"])
        ids = [i for i, meta in zip(data["ids"], data["metadatas"])
               if not (meta or {}).get("source_id")]
        if ids:
            collection.delete(ids=ids)
        logger.info("legacy_file_deleted", source_file=source_file, chunks=len(ids))
        return len(ids)

    def delete_collection(self) -> None:
        """Delete all documents in this collection. Use carefully!"""
        logger.warning("deleting_collection", collection=self.collection_name)
        if hasattr(self._store, "delete_collection"):
            self._store.delete_collection()
        elif hasattr(self._store, "_collection"):
            self._store._collection.delete()

    def get_document_count(self) -> int:
        """Return number of chunks currently in the store."""
        try:
            if settings.vector_store_type == "chroma":
                return self._store._collection.count()
            return -1  # Not available for all backends
        except Exception:
            return -1


# Default singleton vector store instance
vector_store = VectorStore()
