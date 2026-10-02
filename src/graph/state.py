# ============================================================
# src/graph/state.py - LangGraph State Definition
#
# Learning Note:
#   In LangGraph, STATE is the shared data structure that flows
#   through every node. Think of it like a "shopping cart" that
#   gets filled up as the user moves through a checkout process.
#
#   Each node in the graph READS from state and WRITES back to state.
#   The TypedDict annotation ensures type safety at every step.
#
#   Our RAG pipeline state flows like this:
#
#   [Input] -> query_planner -> history_rewriter -> retrieval ->
#   reranker -> context_assembly -> guardrails_input ->
#   generation -> guardrails_output -> evaluation -> [Output]
# ============================================================

from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass


class RAGState(TypedDict, total=False):
    """
    The complete state flowing through the LangGraph RAG pipeline.

    TypedDict with total=False means all fields are optional by default.
    This is important because early nodes don't have values for fields
    that later nodes will fill in.
    """

    # --- Input ---
    original_query: str                    # Raw user query (before any processing)
    session_id: str                        # Unique ID for this conversation session
    user_id: str                           # User identifier (for access control)
    source_filter: Any                     # SourceFilter of sources to search (None = all)

    # --- Query Planning ---
    query_intent: str                      # "factual", "analytical", "conversational"
    planned_query: str                     # Potentially reformulated query
    requires_history: bool                 # Whether to use conversation history

    # --- History-Aware Rewriting ---
    conversation_history: List[Dict]       # List of {"role": "user/assistant", "content": "..."}
    rewritten_query: str                   # Query after history-aware rewriting

    # --- Retrieval ---
    true_data_chunks: List[Dict]           # Relevant chunks (above reranker threshold)
    noisy_data_chunks: List[Dict]          # Filtered chunks (below threshold)
    retrieval_metadata: Dict[str, Any]     # Stats: count, avg_score, etc.

    # --- Context Assembly ---
    assembled_context: str                 # Formatted context string for LLM prompt
    source_documents: List[str]            # Source file names for citation

    # --- Input Guardrails ---
    input_is_safe: bool                    # Whether input passed guardrails
    input_violations: List[str]            # List of violations found
    sanitized_query: str                   # Query after PII redaction

    # --- Generation ---
    llm_response: str                      # Raw LLM response
    model_used: str                        # Which LLM was called

    # --- Output Guardrails ---
    output_is_safe: bool                   # Whether output passed guardrails
    output_violations: List[str]           # Output validation issues
    final_response: str                    # Final sanitized response to user
    hallucination_score: float             # Groundedness score 0-1

    # --- Evaluation ---
    eval_metrics: Dict[str, Any]           # RAGAS metrics dict

    # --- Pipeline Metadata ---
    trace_id: str                          # LangFuse trace ID for observability
    pipeline_steps: List[str]              # Log of completed node names
    error: Optional[str]                   # Error message if pipeline failed
    total_latency_ms: int                  # Total pipeline execution time


def create_initial_state(
    query: str,
    session_id: str = "",
    user_id: str = "anonymous",
    conversation_history: List[Dict] = None,
    source_filter: Any = None,
) -> RAGState:
    """
    Create initial state for a new RAG pipeline run.
    Call this before invoking the graph.
    """
    return RAGState(
        original_query=query,
        session_id=session_id,
        user_id=user_id,
        conversation_history=conversation_history or [],
        source_filter=source_filter,
        pipeline_steps=[],
        input_is_safe=True,
        output_is_safe=True,
        input_violations=[],
        output_violations=[],
        true_data_chunks=[],
        noisy_data_chunks=[],
        source_documents=[],
        eval_metrics={},
        error=None,
    )
