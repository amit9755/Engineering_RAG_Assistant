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
        import hashlib

        collection = self._store._collection
        # Re-indexing mostly sees unchanged text: reuse the stored vector of any chunk
        # whose text is identical (same content hash) instead of embedding it again.
        existing = collection.get(where={"source_id": source_id}, include=["metadatas", "embeddings"])
        old_ids = existing["ids"]
        cached = {}
        for meta, vector in zip(existing.get("metadatas") or [], existing.get("embeddings")
                                if existing.get("embeddings") is not None else []):
            if meta and meta.get("content_hash"):
                cached[meta["content_hash"]] = [float(x) for x in vector]
        model = settings.embedding_model
        for doc in documents:
            doc.metadata["content_hash"] = hashlib.sha1(
                f"{model}\n{doc.page_content}".encode("utf-8")).hexdigest()

        new_ids = [str(uuid.uuid4()) for _ in documents]
        total = len(documents)
        reused = sum(1 for d in documents if d.metadata["content_hash"] in cached)
        logger.info("embedding_cache", source_id=source_id, chunks=total, reused=reused, to_embed=total - reused)
        done = reused
        if on_progress:
            on_progress(done, total)
        try:
            # Chroma rejects oversized batches, and repositories can produce thousands of chunks.
            for start in range(0, total, self.ADD_BATCH_SIZE):
                batch = documents[start:start + self.ADD_BATCH_SIZE]
                ids = new_ids[start:start + self.ADD_BATCH_SIZE]
                missing = [d for d in batch if d.metadata["content_hash"] not in cached]
                if missing:
                    vectors = self._store.embeddings.embed_documents([d.page_content for d in missing])
                    for doc, vector in zip(missing, vectors):
                        cached[doc.metadata["content_hash"]] = vector
                collection.add(ids=ids, documents=[d.page_content for d in batch],
                               metadatas=[d.metadata for d in batch],
                               embeddings=[cached[d.metadata["content_hash"]] for d in batch])
                done += len(missing)
                if on_progress and missing:
                    on_progress(done, total)
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

    def list_files(self, source_ids: List[str] = None) -> List[tuple]:
        """(source_id, source_file, file_path) of every indexed repository file; cached until the index changes."""
        if settings.vector_store_type != "chroma":
            return []
        collection = self._store._collection
        count = collection.count()
        if getattr(self, "_files_cache", (None, None))[0] != count:
            data = collection.get(where={"source_type": "bitbucket"}, include=["metadatas"])
            files = {(m.get("source_id"), m.get("source_file"), m.get("file_path"))
                     for m in data["metadatas"] if m and m.get("file_path") and not m["file_path"].startswith("(")}
            self._files_cache = (count, sorted(files))
        files = self._files_cache[1]
        return [f for f in files if source_ids is None or f[0] in source_ids]

    def get_file_chunks(self, source_id: str, file_path: str, around: int = 0, limit: int = 8) -> List[Document]:
        """Consecutive chunks of one indexed file around a chunk index, in file order."""
        if settings.vector_store_type != "chroma":
            return []
        data = self._store._collection.get(
            where={"$and": [{"source_id": source_id}, {"file_path": file_path}]}, include=["documents", "metadatas"])
        docs = sorted((Document(page_content=t, metadata=m or {}) for t, m in zip(data["documents"], data["metadatas"])),
                      key=lambda d: d.metadata.get("chunk_index", 0))
        if len(docs) <= limit:
            return docs
        start = max(0, min(around - limit // 2, len(docs) - limit))
        return docs[start:start + limit]

    def find_text(self, literal: str, source_filter=None, limit: int = 20) -> List[Document]:
        """Chunks that contain the exact text (like "find in files"), within the selected sources."""
        if settings.vector_store_type != "chroma" or not literal:
            return []
        where = source_filter.chroma_where() if source_filter is not None and not source_filter.is_empty else None
        if source_filter is not None and source_filter.is_empty:
            return []
        data = self._store._collection.get(where=where, where_document={"$contains": literal}, limit=limit,
                                           include=["documents", "metadatas"])
        return [Document(page_content=text, metadata=meta or {})
                for text, meta in zip(data["documents"], data["metadatas"])]

    def has_commit_history(self, source_id: str) -> bool:
        """True if the repository's commit history has been indexed (older indexes lack it)."""
        if settings.vector_store_type != "chroma":
            return False
        found = self._store._collection.get(
            where={"$and": [{"source_id": source_id}, {"file_path": "(commit history)"}]}, limit=1, include=[])
        return bool(found["ids"])

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

    OVERVIEW_DOC_HINTS = ("overview", "architecture", "design", "sdd", "intro", "getting-started",
                          "getting_started", "claude.md", "agents.md", "about")

    def get_overview_chunks(self, source_filter=None, per_source: int = 4) -> List[Document]:
        """
        Chunks that describe each selected source as a whole, for broad questions
        like "explain this project": README and design docs and the file tree of a
        repository, and the opening chunks of a document or older upload.
        """
        if settings.vector_store_type != "chroma":
            return []
        collection = self._store._collection
        where = {"chunk_index": {"$lt": 2}}
        if source_filter is not None:
            if source_filter.is_empty:
                return []
            where = {"$and": [where, source_filter.chroma_where()]}
        data = collection.get(where=where, include=["metadatas"])

        def priority(meta):
            path = (meta.get("file_path") or "").lower()
            name = path.rsplit("/", 1)[-1]
            depth = path.count("/")
            if meta.get("source_type") == "bitbucket":
                if name.startswith("readme"):
                    return (0, depth, meta.get("chunk_index", 0))
                if any(hint in path for hint in self.OVERVIEW_DOC_HINTS) and name.endswith((".md", ".rst", ".txt")):
                    return (1, depth, meta.get("chunk_index", 0))
                if path == "(file tree)" and meta.get("chunk_index", 0) == 0:
                    return (2, 0, 0)
                return None
            if meta.get("source_type") == "jira":
                return None
            if meta.get("source_type") == "confluence":
                # the space's top-level pages (home page first) describe it best
                return (meta.get("depth", 9), 0, meta.get("chunk_index", 0)) if meta.get("depth", 9) <= 1 else None
            return (0, 0, meta.get("chunk_index", 0))  # documents and older uploads: their opening

        groups = {}
        for chunk_id, meta in zip(data["ids"], data["metadatas"]):
            meta = meta or {}
            rank = priority(meta)
            if rank is not None:
                key = meta.get("source_id") or meta.get("source_file")
                groups.setdefault(key, []).append((rank, chunk_id))
        chosen = [cid for items in groups.values() for _, cid in sorted(items)[:per_source]]
        if not chosen:
            return []
        picked = collection.get(ids=chosen, include=["documents", "metadatas"])
        order = {cid: i for i, cid in enumerate(chosen)}
        docs = sorted(zip(picked["ids"], picked["documents"], picked["metadatas"]), key=lambda x: order[x[0]])
        return [Document(page_content=text, metadata=meta) for _, text, meta in docs]

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
