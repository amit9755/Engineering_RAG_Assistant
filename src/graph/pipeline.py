# ============================================================
# src/graph/pipeline.py - LangGraph Pipeline Assembly
#
# Learning Note:
#   This file WIRES all nodes together into a directed graph.
#   Think of it like drawing arrows on a flowchart diagram.
#
#   LangGraph concepts:
#     StateGraph  - the graph container
#     add_node()  - register a node function with a name
#     add_edge()  - unconditional arrow from node A to node B
#     add_conditional_edges() - arrow that branches based on a function
#     set_entry_point() - which node runs first
#     set_finish_point() - which node(s) can end the graph
#     compile() - validate and build the runnable graph
#
#   GRAPH STRUCTURE:
#
#   START
#     |
#   [input_guardrails]
#     |
#     +-- (blocked) --> [error_response] --> END
#     |
#   (continue)
#     |
#   [query_planner]
#     |
#   [history_rewriter]
#     |
#   [retrieval]
#     |
#   [context_assembly]
#     |
#   [generation]
#     |
#   [output_guardrails]
#     |
#   [evaluation]
#     |
#   END
# ============================================================

from langgraph.graph import StateGraph, END, START
from src.graph.state import RAGState
from src.graph.nodes import (
    input_guardrails_node,
    query_planner_node,
    history_rewriter_node,
    retrieval_node,
    context_assembly_node,
    generation_node,
    output_guardrails_node,
    evaluation_node,
    error_response_node,
    route_after_input_guardrails,
)
from src.observability.logger import get_logger

logger = get_logger(__name__)


def build_rag_pipeline() -> StateGraph:
    """
    Build and compile the complete RAG LangGraph pipeline.

    Returns a compiled graph that can be invoked with:
        result = graph.invoke(initial_state)

    Learning Note:
        compile() validates the graph - it checks for:
        - Nodes with no incoming edges (dead ends)
        - Missing conditional edge targets
        - Type mismatches in state
        Always call compile() before using the graph.
    """
    # Create the state graph with our RAGState type
    graph = StateGraph(RAGState)

    # --- Register all nodes ---
    # The string name is what you use in add_edge() calls
    graph.add_node("input_guardrails", input_guardrails_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("history_rewriter", history_rewriter_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("context_assembly", context_assembly_node)
    graph.add_node("generation", generation_node)
    graph.add_node("output_guardrails", output_guardrails_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("error_response", error_response_node)

    # --- Set entry point (first node to run) ---
    graph.set_entry_point("input_guardrails")

    # --- Add conditional edge after input guardrails ---
    # route_after_input_guardrails() returns "continue" or "blocked"
    # We map those strings to node names
    graph.add_conditional_edges(
        "input_guardrails",
        route_after_input_guardrails,
        {
            "continue": "query_planner",    # safe input -> continue
            "blocked": "error_response",     # unsafe input -> error
        },
    )

    # --- Add unconditional edges for the happy path ---
    graph.add_edge("query_planner", "history_rewriter")
    graph.add_edge("history_rewriter", "retrieval")
    graph.add_edge("retrieval", "context_assembly")
    graph.add_edge("context_assembly", "generation")
    graph.add_edge("generation", "output_guardrails")
    graph.add_edge("output_guardrails", "evaluation")

    # --- Set finish points ---
    graph.add_edge("evaluation", END)
    graph.add_edge("error_response", END)

    # Compile validates the graph structure
    compiled = graph.compile()
    logger.info("rag_pipeline_compiled")

    return compiled


# ============================================================
# Pipeline Runner - convenience wrapper around the graph
# ============================================================

class RAGPipeline:
    """
    High-level interface to run the RAG pipeline.
    Wraps the compiled LangGraph with tracing and error handling.
    """

    def __init__(self):
        self._graph = build_rag_pipeline()
        logger.info("rag_pipeline_ready")

    def run(
        self,
        query: str,
        session_id: str = "",
        user_id: str = "anonymous",
        conversation_history: list = None,
        source_filter=None,
    ) -> dict:
        """
        Run the full RAG pipeline for a user query.

        Args:
            query: User's question
            session_id: Conversation session identifier
            user_id: User identifier for access control
            conversation_history: List of prior messages
            source_filter: SourceFilter restricting retrieval (None = all sources)

        Returns:
            dict with: answer, eval_metrics, true_data_chunks,
                       noisy_data_chunks, pipeline_steps, sources
        """
        import time
        from src.graph.state import create_initial_state
        from src.observability.tracer import tracer

        start_time = time.time()

        # Create trace for observability
        trace = tracer.create_trace(
            name="rag_pipeline",
            user_id=user_id,
            session_id=session_id,
            metadata={"query_length": len(query)},
        )

        # Build initial state
        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            conversation_history=conversation_history or [],
            source_filter=source_filter,
        )

        try:
            # Execute the graph
            final_state = self._graph.invoke(initial_state)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                "pipeline_complete",
                session_id=session_id,
                latency_ms=elapsed_ms,
                steps=final_state.get("pipeline_steps", []),
            )

            # Log eval scores to LangFuse
            eval_metrics = final_state.get("eval_metrics", {})
            if eval_metrics.get("evaluated"):
                tracer.log_eval_scores(trace, {
                    k: v for k, v in eval_metrics.items()
                    if isinstance(v, float)
                })

            # Build clean response dict
            return {
                "answer": final_state.get("final_response", "No response generated."),
                "sources": final_state.get("source_documents", []),
                "true_data_chunks": final_state.get("true_data_chunks", []),
                "noisy_data_chunks": final_state.get("noisy_data_chunks", []),
                "eval_metrics": eval_metrics,
                "pipeline_steps": final_state.get("pipeline_steps", []),
                "query_intent": final_state.get("query_intent", "factual"),
                "rewritten_query": final_state.get("rewritten_query", query),
                "hallucination_score": final_state.get("hallucination_score", 0.0),
                "model_used": final_state.get("model_used", "unknown"),
                "latency_ms": elapsed_ms,
                "input_safe": final_state.get("input_is_safe", True),
                "output_safe": final_state.get("output_is_safe", True),
                "error": final_state.get("error"),
            }

        except Exception as exc:
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.error("pipeline_failed", error=str(exc), latency_ms=elapsed_ms)
            return {
                "answer": "An internal error occurred. Please try again.",
                "sources": [],
                "true_data_chunks": [],
                "noisy_data_chunks": [],
                "eval_metrics": {},
                "pipeline_steps": [],
                "latency_ms": elapsed_ms,
                "error": str(exc),
            }


# Singleton pipeline instance - built once, reused for all requests
rag_pipeline = RAGPipeline()
