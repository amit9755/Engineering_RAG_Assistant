# ============================================================
# src/observability/tracer.py - LangFuse Tracing Integration
#
# Learning Note:
#   LangFuse is an open-source LLM observability platform.
#   It traces every LangGraph node execution, capturing:
#     - Input/output of each node
#     - Latency per node
#     - Token usage and cost
#     - Evaluation scores
#   This is critical for debugging and improving RAG quality.
#   Think of it like "distributed tracing" (Jaeger/Zipkin)
#   but purpose-built for LLM applications.
# ============================================================

from typing import Optional
from src.config import settings
from src.observability.logger import get_logger

logger = get_logger(__name__)


class RAGTracer:
    """
    Wraps LangFuse for tracing RAG pipeline executions.
    Each user request becomes a 'trace', and each LangGraph
    node execution becomes a 'span' within that trace.
    """

    def __init__(self):
        self._langfuse = None
        self._enabled = False
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Initialize LangFuse client if credentials are configured."""
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                from langfuse import Langfuse
                self._langfuse = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                self._enabled = True
                logger.info("langfuse_initialized", host=settings.langfuse_host)
            except ImportError:
                logger.warning("langfuse_not_installed", msg="pip install langfuse to enable tracing")
            except Exception as exc:
                logger.warning("langfuse_init_failed", error=str(exc))
        else:
            logger.info("langfuse_disabled", reason="No credentials configured")

    def create_trace(self, name: str, user_id: str = "anonymous", session_id: str = "", metadata: dict = None):
        """
        Start a new trace for a user request.
        Returns a trace object or None if tracing is disabled.

        Learning Note:
            A 'trace' is the top-level container for one user request.
            All node spans are nested under it.
        """
        if not self._enabled:
            return None
        try:
            return self._langfuse.trace(
                name=name,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning("trace_creation_failed", error=str(exc))
            return None

    def log_span(
        self,
        trace,
        name: str,
        input_data: dict,
        output_data: dict,
        metadata: dict = None,
    ) -> None:
        """
        Log a single node execution as a span within a trace.
        Call this at the end of each LangGraph node.
        """
        if not self._enabled or trace is None:
            return
        try:
            trace.span(
                name=name,
                input=input_data,
                output=output_data,
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning("span_log_failed", node=name, error=str(exc))

    def log_eval_scores(self, trace, scores: dict) -> None:
        """
        Attach evaluation scores (faithfulness, relevancy, etc.)
        to the trace so they appear in the LangFuse dashboard.
        """
        if not self._enabled or trace is None:
            return
        try:
            for metric_name, value in scores.items():
                trace.score(name=metric_name, value=value)
        except Exception as exc:
            logger.warning("eval_score_log_failed", error=str(exc))

    def flush(self) -> None:
        """Flush pending events to LangFuse (call on shutdown)."""
        if self._enabled and self._langfuse:
            try:
                self._langfuse.flush()
            except Exception:
                pass


# Singleton tracer instance
tracer = RAGTracer()
