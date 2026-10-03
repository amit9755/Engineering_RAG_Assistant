# ============================================================
# src/retrieval/hybrid_retriever.py - Hybrid Retrieval + Re-Ranking
#
# Learning Note:
#   WHY HYBRID RETRIEVAL?
#   Vector (dense) search is great for semantic similarity but
#   can miss exact keyword matches. BM25 (sparse) search is great
#   for exact keyword matching but misses paraphrases.
#
#   Example: Query = "What is the BERT acronym?"
#     - Vector search: finds chunks about BERT transformers (semantic)
#     - BM25 search:   finds chunks containing the exact word "BERT" (keyword)
#     - HYBRID:        combines both = best coverage!
#
#   RECIPROCAL RANK FUSION (RRF):
#     Combines ranked lists from multiple retrievers.
#     Formula: RRF_score = sum(1 / (k + rank_i)) for each retriever
#     k=60 is the standard constant from the original RRF paper.
#     This is better than simply averaging scores because it is
#     robust to score scale differences between retrievers.
#
#   RE-RANKING (True Data vs Noisy Data):
#     After hybrid retrieval, we have ~10-20 candidate chunks.
#     A Cross-Encoder re-ranker scores each (query, chunk) pair
#     together (unlike bi-encoders that score them separately).
#     Chunks below RERANKER_THRESHOLD are "Noisy Data" - filtered out.
#     Chunks above threshold are "True Data" - sent to the LLM.
# ============================================================

import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
from langchain_core.documents import Document
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ScoredChunk:
    """A document chunk with its relevance score and classification."""
    document: Document
    score: float
    is_true_data: bool        # True if above reranker_threshold
    retrieval_method: str     # "vector", "bm25", or "hybrid"


_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """
    Lowercase words, splitting paths, punctuation and camelCase, so
    "lib/instagram/publish.ts" and "publishToInstagram" both match the
    query word "instagram" (whitespace splitting would not).
    """
    return _WORD.findall(_CAMEL.sub(r"\1 \2", text).lower())


class BM25Retriever:
    """
    BM25 (Best Match 25) lexical retriever.

    Learning Note:
        BM25 is a classic information retrieval algorithm (used by
        Elasticsearch by default). It scores documents based on:
        - Term frequency (how often the query word appears)
        - Inverse document frequency (rare words score higher)
        - Document length normalization

        No neural network needed - fast and explainable!
    """

    def __init__(self, documents: List[Document] = None):
        self._retriever = None
        self._documents = documents or []
        if documents:
            self._build_index(documents)

    def _build_index(self, documents: List[Document]) -> None:
        """Build BM25 index from a list of documents."""
        try:
            from rank_bm25 import BM25Okapi
            # Tokenise each document by splitting on whitespace
            # In production, use a proper tokenizer (NLTK, spaCy)
            tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
            self._retriever = BM25Okapi(tokenized_corpus)
            self._documents = documents
            logger.info("bm25_index_built", doc_count=len(documents))
        except Exception as exc:
            logger.error("bm25_index_failed", error=str(exc))

    def update_index(self, documents: List[Document]) -> None:
        """Rebuild BM25 index with new documents."""
        if not documents:
            self._documents = []
            self._retriever = None
            return
        self._build_index(documents)

    def search(self, query: str, k: int = None, source_filter=None) -> List[Tuple[Document, float]]:
        """
        Search BM25 index and return top-k (document, score) pairs.
        Scores are BM25 relevance scores (higher = more relevant).
        source_filter restricts results to the selected sources (None = all).
        """
        if not self._retriever:
            logger.warning("bm25_search_called_without_index")
            return []

        k = k or settings.retrieval_top_k
        tokenized_query = tokenize(query)
        scores = self._retriever.get_scores(tokenized_query)

        # Pair documents with their scores and sort
        scored_pairs = list(zip(self._documents, scores))
        if source_filter is not None:
            scored_pairs = [p for p in scored_pairs if source_filter.matches(p[0].metadata)]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)
        return scored_pairs[:k]


class SemanticReRanker:
    """
    Cross-Encoder re-ranker that scores (query, document) pairs jointly.

    Learning Note:
        Bi-Encoder (used in embedding/vector search):
            encodes query and document SEPARATELY, then compares vectors.
            Fast but less accurate for relevance scoring.

        Cross-Encoder (used here):
            encodes query AND document TOGETHER in one pass.
            Much more accurate - the model can attend between both texts.
            Slower (can't pre-compute) but perfect for re-ranking a small set.

        Workflow:
            1. Fast bi-encoder retrieves top-20 candidates (cheap)
            2. Slow cross-encoder re-ranks the 20 (expensive but small set)
            3. Filter by threshold -> True Data vs Noisy Data
    """

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            # Disable SSL verification for corporate proxy environments
            from src.retrieval.reranker_offline import disable_ssl_for_hf
            disable_ssl_for_hf()
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                settings.reranker_model,
                max_length=512,
            )
            logger.info("reranker_loaded", model=settings.reranker_model)
        except Exception as exc:
            logger.warning("reranker_load_failed", error=str(exc), msg="Re-ranking disabled")
            self._model = None

    def rerank(
        self,
        query: str,
        documents: List[Document],
    ) -> List[ScoredChunk]:
        """
        Score each document against the query and classify as
        True Data (relevant) or Noisy Data (irrelevant).

        Returns list of ScoredChunk sorted by score descending.
        """
        if not documents:
            return []

        if self._model is None:
            # Fallback: return all as True Data without re-ranking
            logger.warning("reranker_unavailable_fallback")
            return [ScoredChunk(doc, 1.0, True, "fallback") for doc in documents]

        # Build (query, chunk_text) pairs for the cross-encoder
        pairs = [(query, doc.page_content) for doc in documents]

        # Score all pairs in one batch call (efficient)
        scores = self._model.predict(pairs)

        # Apply sigmoid to convert logits to 0-1 probability range
        import numpy as np
        scores_normalized = 1 / (1 + np.exp(-scores))

        scored_chunks = []
        for doc, raw_score, norm_score in zip(documents, scores, scores_normalized):
            is_true = float(norm_score) >= settings.reranker_threshold
            scored_chunks.append(
                ScoredChunk(
                    document=doc,
                    score=float(norm_score),
                    is_true_data=is_true,
                    retrieval_method="reranked",
                )
            )

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x.score, reverse=True)

        true_data = [c for c in scored_chunks if c.is_true_data]
        noisy_data = [c for c in scored_chunks if not c.is_true_data]

        logger.info(
            "reranking_complete",
            total_candidates=len(scored_chunks),
            true_data=len(true_data),
            noisy_data=len(noisy_data),
            threshold=settings.reranker_threshold,
        )

        return scored_chunks


# Questions about commits are answered from the indexed commit history. The
# cross-encoder judges that history poorly ("last 5 commits" scores ~0, "latest
# 5 commits" ~1), so it is included directly instead of relying on its score.
COMMIT_INTENT = re.compile(
    r"\b(commits?|committed|git log|commit history|recent changes|latest changes|changelog|"
    r"who (changed|pushed|modified)|last (change|push|update)s?)\b", re.I)


# Broad questions ("explain this project", "give me an overview") rarely match a
# specific chunk, so the cross-encoder rejects everything. They always get the
# overview material of the selected sources instead (README / design docs / file
# tree of a repository, the opening of a document).
_OVERVIEW_STRONG = re.compile(
    r"\b(overview|summar(y|ise|ize)|architecture|walk me through|high[- ]level|tell me about|introduc\w*|"
    r"purpose of|what is this|what's this|whats this)\b", re.I)
_OVERVIEW_WEAK = re.compile(r"\b(explain|describe|what (is|does)|what's|whats|how does)\b", re.I)
_OVERVIEW_OBJECT = re.compile(r"\b(this|the|whole|entire) (project|repo|repository|code ?base|app|application|"
                              r"system|product|service|software|tool|document)\b|\b(this|it)\b\W*$", re.I)


def is_overview_question(text: str) -> bool:
    text = (text or "").strip()
    if _OVERVIEW_STRONG.search(text):
        return True
    if not _OVERVIEW_WEAK.search(text):
        return False
    match = _OVERVIEW_OBJECT.search(text)
    if not match:
        return False
    # "explain this" / "describe it" are broad only when that is the whole question;
    # "how do I fix it?" ends in "it" but is about something specific.
    return bool(match.group(2)) or len(text.split()) <= 5


# Code names in a question (fetchDeviceDetails, InstagramPanel, overall_status,
# INSTAGRAM_RATE_LIMITED, api.client). Meaning / keyword search splits them into
# common words, so they are also looked up word-for-word.
_CODE_NAME = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|"
                        r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z_]\w*\.[a-z_]\w*(?:\.[a-z_]\w*)*)\b")
_FILE_EXT = re.compile(r"\.(?:ts|tsx|js|jsx|py|java|go|rb|php|cs|html|css|scss|json|ya?ml|md|sql)$", re.I)


def code_names(text: str, limit: int = 5) -> List[str]:
    names = []
    for match in _CODE_NAME.finditer(text or ""):
        name = match.group(0)
        if not _FILE_EXT.search(name) and name not in names and len(name) >= 4:
            names.append(name)
    return names[:limit]


def _is_definition(text: str, name: str) -> bool:
    """A line that defines the name (function / class / const / method / object key / assignment)."""
    n = re.escape(name)
    return bool(re.search(
        rf"(?:\b(?:function|def|class|const|let|var|interface|type|enum|async)\s+{n}\b|"
        rf"^\s*(?:export\s+)?(?:public|private|protected|static|readonly|\s)*{n}\s*[:=(]|"
        rf"^\s*['\"]?{n}['\"]?\s*:)", text, re.M))


def split_questions(text: str) -> List[str]:
    """Split a message into separate questions (one per line or per '?')."""
    import re
    parts = []
    for line in text.splitlines():
        parts += [p.strip() for p in re.split(r"(?<=\?)\s+", line)]
    questions = [p for p in parts if len(p.split()) >= 3]
    return questions if len(questions) > 1 else [text.strip()]


def _chunk_key(doc: Document) -> str:
    return doc.metadata.get("chunk_id") or doc.metadata.get("document_id") or doc.page_content[:50]


class HybridRetriever:
    """
    Full hybrid retrieval pipeline:
        1. Dense vector search (ChromaDB / Vertex AI)
        2. Sparse BM25 keyword search
        3. Reciprocal Rank Fusion to merge results
        4. Cross-encoder re-ranking
        5. Noise filtering (True Data vs Noisy Data)
    """

    def __init__(self):
        from src.retrieval.vector_store import vector_store
        self.vector_store = vector_store
        self.bm25 = BM25Retriever()
        self.reranker = SemanticReRanker()
        self._bm25_ready = False

    def update_bm25_index(self, documents: List[Document]) -> None:
        """Rebuild BM25 index after new documents are ingested."""
        self.bm25.update_index(documents)
        self._bm25_ready = True

    def _ensure_bm25(self) -> None:
        """The BM25 index lives in memory: build it from the vector store on first use after a restart."""
        if not self._bm25_ready:
            from src.ingestion.ingestion_pipeline import ingestion_pipeline
            ingestion_pipeline._refresh_bm25_index()
            self._bm25_ready = True

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Document],
        bm25_results: List[Tuple[Document, float]],
        k: int = 60,
    ) -> List[Document]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        Learning Note:
            RRF formula: score(doc) = sum over retrievers of 1/(k + rank)
            k=60 prevents top-ranked documents from dominating.
            Documents appearing in BOTH lists get higher scores.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        # Score from vector search
        for rank, doc in enumerate(vector_results, start=1):
            doc_id = _chunk_key(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        # Score from BM25
        for rank, (doc, _bm25_score) in enumerate(bm25_results, start=1):
            doc_id = _chunk_key(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            doc_map[doc_id] = doc

        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids]

    MAX_SUB_QUESTIONS = 6
    MAX_MULTI_CHUNKS = 12

    def code_name_chunks(self, query: str, source_filter=None, per_name: int = 4) -> List[ScoredChunk]:
        """Chunks that contain code names from the question verbatim: definitions first, then uses."""
        found, seen = [], set()
        for name in code_names(query):
            try:
                hits = self.vector_store.find_text(name, source_filter, limit=40)
            except Exception as exc:
                logger.warning("code_name_lookup_failed", name=name, error=str(exc)[:120])
                continue
            hits = [h for h in hits if not (h.metadata.get("file_path") or "").startswith("(")]
            definitions = [h for h in hits if _is_definition(h.page_content, name)]
            uses = [h for h in hits if h not in definitions]
            for doc in definitions[:2] + uses[:max(1, per_name - len(definitions[:2]))]:
                key = _chunk_key(doc)
                if key not in seen:
                    seen.add(key)
                    found.append(ScoredChunk(doc, 1.0, True, "code-name"))
        if found:
            logger.info("code_names_found", names=code_names(query), chunks=len(found))
        return found

    def retrieve(
        self,
        query: str,
        k: int = None,
        source_filter=None,
    ) -> Tuple[List[ScoredChunk], List[ScoredChunk]]:
        """
        Retrieve for a query; a message containing several questions is
        split so each question gets its own search and its own chunks
        (one search for five topics returns chunks for only one or two).
        Code names in the question are also looked up word-for-word and put first.
        """
        self._ensure_bm25()
        true_data, noisy_data = self._retrieve_split(query, k, source_filter)
        exact = self.code_name_chunks(query, source_filter)
        if exact:
            keys = {_chunk_key(c.document) for c in exact}
            true_data = exact + [c for c in true_data if _chunk_key(c.document) not in keys]
            true_data = true_data[:max(settings.final_top_k, len(exact)) + 4]
            noisy_data = [c for c in noisy_data if _chunk_key(c.document) not in keys]
        return true_data, noisy_data

    def _retrieve_split(self, query, k, source_filter):
        questions = split_questions(query)[:self.MAX_SUB_QUESTIONS]
        if len(questions) <= 1:
            return self._retrieve_one(query, k, source_filter, settings.final_top_k)

        per_question = max(2, self.MAX_MULTI_CHUNKS // len(questions))
        true_data, noisy_data, seen = [], [], set()
        for question in questions:
            q_true, q_noisy = self._retrieve_one(question, k, source_filter, per_question)
            for chunk in q_true:
                key = _chunk_key(chunk.document)
                if key not in seen:
                    seen.add(key)
                    true_data.append(chunk)
            noisy_data += q_noisy
        logger.info("multi_question_retrieval", questions=len(questions), true_data=len(true_data))
        return true_data, noisy_data

    def _retrieve_one(
        self,
        query: str,
        k: int,
        source_filter,
        limit: int,
    ) -> Tuple[List[ScoredChunk], List[ScoredChunk]]:
        """
        Full retrieval pipeline for a query.

        source_filter (SourceFilter or None) limits both vector and BM25
        results to the sources selected in the UI; None searches everything.

        Returns:
            (true_data_chunks, noisy_data_chunks)
            true_data_chunks: relevant chunks to send to LLM
            noisy_data_chunks: filtered out chunks (for UI visibility)
        """
        k = k or settings.retrieval_top_k

        # Step 1: Dense vector search
        vector_results = self.vector_store.similarity_search_with_filter(query, source_filter, k=k)
        logger.info("vector_retrieval", results=len(vector_results))

        # Step 2: BM25 keyword search
        bm25_results = self.bm25.search(query, k=k, source_filter=source_filter)
        logger.info("bm25_retrieval", results=len(bm25_results))

        # Step 3: Merge with Reciprocal Rank Fusion
        merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
        logger.info("rrf_merged", total=len(merged))

        # Step 4: Cross-encoder re-ranking + noise filtering
        scored_chunks = self.reranker.rerank(query, merged)

        # Step 5: Split True Data vs Noisy Data
        true_data = [c for c in scored_chunks if c.is_true_data][:limit]
        noisy_data = [c for c in scored_chunks if not c.is_true_data]

        # Step 6: broad questions always get each source's overview material
        if is_overview_question(query):
            overview = self.vector_store.get_overview_chunks(source_filter)
            keys = {_chunk_key(c.document) for c in true_data}
            extra = [ScoredChunk(d, 1.0, True, "overview") for d in overview if _chunk_key(d) not in keys]
            true_data = (true_data + extra)[:max(limit, 8)]
            noisy_data = [c for c in noisy_data if _chunk_key(c.document) not in {_chunk_key(d) for d in overview}]
            logger.info("overview_chunks_included", chunks=len(extra))

        # Step 7: commit questions always get the newest commit history
        if COMMIT_INTENT.search(query):
            history = self.vector_store.get_commit_history(source_filter)
            keys = {_chunk_key(d) for d in history}
            true_data = ([ScoredChunk(d, 1.0, True, "commit-history") for d in history] +
                         [c for c in true_data if _chunk_key(c.document) not in keys])[:max(limit, len(history))]
            noisy_data = [c for c in noisy_data if _chunk_key(c.document) not in keys]
            logger.info("commit_history_included", chunks=len(history))

        return true_data, noisy_data


# Singleton instance
hybrid_retriever = HybridRetriever()
