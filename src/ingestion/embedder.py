# ============================================================
# src/ingestion/embedder.py - Embedding Model Wrapper
#
# Learning Note:
#   Embeddings convert text into dense numerical vectors (arrays
#   of floats). Similar texts produce vectors that are "close"
#   in vector space (measured by cosine similarity).
#
#   Example:
#     "machine learning" -> [0.12, -0.34, 0.89, ...]  (384 dims)
#     "deep learning"    -> [0.11, -0.31, 0.91, ...]  (similar!)
#     "football match"   -> [-0.71, 0.23, -0.12, ...]  (different)
#
#   We support two backends:
#     - sentence-transformers: local, free, good for dev/testing
#     - Vertex AI text-embedding: Google's model, better quality,
#       needs GCP credentials, costs money
# ============================================================

import os
from typing import List
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """
    Unified embedding interface that supports both local
    HuggingFace models and Google Vertex AI embeddings.
    """

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """
        Load the appropriate embedding model based on config.
        Learning Note: We lazy-load so the app starts fast and
        only loads the model when first needed.
        """
        model_name = settings.embedding_model

        if "textembedding" in model_name or "gecko" in model_name:
            # Google Vertex AI embeddings - production grade
            self._load_vertex_embeddings(model_name)
        else:
            # HuggingFace sentence-transformers - local/dev
            self._load_local_embeddings(model_name)

    def _load_local_embeddings(self, model_name: str) -> None:
        """Load a local sentence-transformers model."""
        try:
            # Disable SSL for corporate proxy environments
            from src.retrieval.reranker_offline import disable_ssl_for_hf
            disable_ssl_for_hf()
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            # Embedding is CPU-bound during indexing: use all cores but one (PyTorch
            # often defaults to fewer), leaving one free so the app stays responsive.
            try:
                import torch
                threads = int(os.environ.get("EMBEDDING_THREADS", "0")) or max(1, (os.cpu_count() or 2) - 1)
                torch.set_num_threads(threads)
                logger.info("embedding_threads", threads=threads)
            except Exception as exc:
                logger.warning("embedding_threads_not_set", error=str(exc))
            self._model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # use "cuda" if GPU available
                encode_kwargs={"normalize_embeddings": True},  # normalise for cosine similarity
            )
            logger.info("local_embeddings_loaded", model=model_name)
        except Exception as exc:
            logger.error("local_embeddings_failed", model=model_name, error=str(exc))
            raise

    def _load_vertex_embeddings(self, model_name: str) -> None:
        """Load Google Vertex AI embeddings."""
        try:
            from langchain_google_vertexai import VertexAIEmbeddings
            self._model = VertexAIEmbeddings(
                model_name=model_name,
                project=settings.gcp_project_id,
                location=settings.gcp_region,
            )
            logger.info("vertex_embeddings_loaded", model=model_name)
        except Exception as exc:
            logger.error("vertex_embeddings_failed", model=model_name, error=str(exc))
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string. Used during retrieval.

        Learning Note:
            Query embedding must use the SAME model as document
            embedding. If you index with model A and query with
            model B, results will be garbage.
        """
        return self._model.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts in batch. Used during ingestion.
        Batch processing is more efficient than one-by-one.
        """
        return self._model.embed_documents(texts)

    @property
    def langchain_model(self):
        """Return the raw LangChain embeddings object (for vector stores)."""
        return self._model


# Singleton instance
embedding_model = EmbeddingModel()
