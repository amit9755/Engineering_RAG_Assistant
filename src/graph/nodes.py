# ============================================================
# src/graph/nodes.py - LangGraph Node Implementations
#
# Learning Note:
#   Each FUNCTION here is one node in the LangGraph pipeline.
#   A node receives the current RAGState, does its work,
#   and returns a PARTIAL state update (only the fields it changed).
#   LangGraph merges this partial update back into the full state.
#
#   Node execution order is defined in pipeline.py via add_edge().
#   Conditional edges allow branching (e.g., skip generation if unsafe).
# ============================================================

import time
from typing import Dict, Any, List

from src.graph.state import RAGState
from src.observability.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# NODE 1: Input Guardrails
# First line of defence - validate the user's input
# ============================================================
def input_guardrails_node(state: RAGState) -> Dict[str, Any]:
    """
    Validate and sanitize user input before processing.

    Learning Note:
        This is the first node. If input is unsafe (injection attack,
        empty, etc.) we set input_is_safe=False. The conditional edge
        after this node checks that flag and routes to an error response
        instead of continuing the pipeline.
    """
    logger.info("node_input_guardrails_start")
    from src.guardrails.input_guardrails import input_guardrails

    query = state.get("original_query", "")
    result = input_guardrails.validate(query)

    steps = state.get("pipeline_steps", [])
    steps.append("input_guardrails")

    return {
        "input_is_safe": result.is_safe,
        "input_violations": result.violations,
        "sanitized_query": result.sanitized_text,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 2: Query Planner
# Understand the intent of the query to guide retrieval strategy
# ============================================================
def query_planner_node(state: RAGState) -> Dict[str, Any]:
    """
    Classify the query intent and decide retrieval strategy.

    Learning Note:
        QUERY PLANNING is important for complex queries.
        "What is BERT?" -> factual, use direct retrieval
        "Compare BERT vs GPT" -> analytical, may need multiple retrievals
        "What did we discuss earlier?" -> conversational, needs history

        For simple systems you might skip this. For enterprise RAG,
        this node can also decompose multi-part questions.
    """
    logger.info("node_query_planner_start")

    query = state.get("sanitized_query") or state.get("original_query", "")
    history = state.get("conversation_history", [])

    # Simple rule-based intent classification
    # In production: use LLM-based classification
    query_lower = query.lower()

    if any(word in query_lower for word in ["compare", "difference", "vs", "versus", "contrast"]):
        intent = "analytical"
    elif any(word in query_lower for word in ["earlier", "before", "previous", "you said", "we discussed"]):
        intent = "conversational"
    elif any(word in query_lower for word in ["summarize", "summary", "overview"]):
        intent = "summary"
    else:
        intent = "factual"

    requires_history = intent == "conversational" and len(history) > 0

    steps = state.get("pipeline_steps", [])
    steps.append("query_planner")

    logger.info("query_planned", intent=intent, requires_history=requires_history)

    return {
        "query_intent": intent,
        "planned_query": query,
        "requires_history": requires_history,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 3: History-Aware Query Rewriter
# Reformulate query using conversation context
# ============================================================
def history_rewriter_node(state: RAGState) -> Dict[str, Any]:
    """
    Rewrite the query incorporating conversation history context.

    Learning Note:
        Without history-awareness, a follow-up question like
        "What are its limitations?" has no clear referent.
        This node uses the LLM to rewrite it into a self-contained
        question: "What are the limitations of BERT?"

        This technique is called CONTEXTUAL QUERY COMPRESSION.
        It is essential for multi-turn conversational RAG.
    """
    logger.info("node_history_rewriter_start")

    query = state.get("planned_query", state.get("original_query", ""))
    history = state.get("conversation_history", [])
    requires_history = state.get("requires_history", False)

    steps = state.get("pipeline_steps", [])
    steps.append("history_rewriter")

    # Only rewrite if there's history and it's needed
    if not requires_history or not history:
        return {
            "rewritten_query": query,
            "pipeline_steps": steps,
        }

    try:
        from src.gateway.llm_gateway import llm_gateway

        # Build history context string
        history_str = "\n".join([
            f"{msg['role'].title()}: {msg['content']}"
            for msg in history[-6:]  # last 3 turns
        ])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter. Given a conversation history and a follow-up question, "
                    "rewrite the question to be fully self-contained (no pronouns referring to prior context). "
                    "Output ONLY the rewritten question, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"Conversation:\n{history_str}\n\nFollow-up question: {query}\n\nRewritten question:",
            },
        ]

        rewritten = llm_gateway.complete(messages, temperature=0.0, max_tokens=200)
        logger.info("query_rewritten", original=query[:50], rewritten=rewritten[:50])

        return {"rewritten_query": rewritten.strip(), "pipeline_steps": steps}

    except Exception as exc:
        logger.warning("history_rewrite_failed", error=str(exc))
        return {"rewritten_query": query, "pipeline_steps": steps}


SOURCE_TYPE_LABELS = {"document": "document", "bitbucket": "code", "jira": "Jira issue"}

# Shared by the graph's generation node and the streaming endpoint.
SYSTEM_PROMPT = """You are a precise engineering assistant. Answer using ONLY the numbered
sources below. They come from the user's uploaded documents, Bitbucket repository
files, and Jira issues.

How to answer:
- If the user asks several questions, answer each one under its own short heading.
- Explain in your own words, then cite where it comes from in parentheses using the
  file path or Jira key, e.g. (lib/instagram/publish.ts) or (BT-12).
- For code, name the files, functions, and steps involved; quote a few key lines when useful.
- If the sources do not cover a question, say so for that question only and answer the rest.
- Never invent files, functions, or behaviour that the sources do not show.

Sources:
{context}"""


def format_context(chunks: List[Dict]) -> str:
    """Number each chunk and label its source so the LLM can cite it."""
    parts = []
    for idx, chunk in enumerate(chunks, 1):
        kind = SOURCE_TYPE_LABELS.get(chunk.get("source_type"), "document")
        parts.append(f"[{idx}] ({kind}) {chunk.get('source', 'unknown')}\n{chunk.get('content', '')}")
    return "\n\n".join(parts)


# ============================================================
# NODE 4: Hybrid Retrieval
# Vector + BM25 + Re-ranking + True/Noisy Data classification
# ============================================================
def retrieval_node(state: RAGState) -> Dict[str, Any]:
    """
    Retrieve relevant document chunks using hybrid search.

    Learning Note:
        This is the CORE of RAG. The quality of retrieval directly
        determines the quality of the final answer.
        We use: vector search + BM25 + RRF fusion + cross-encoder re-ranking
        The re-ranker CLASSIFIES chunks as True Data vs Noisy Data.
    """
    logger.info("node_retrieval_start")
    from src.retrieval.hybrid_retriever import hybrid_retriever

    query = state.get("rewritten_query") or state.get("planned_query") or state.get("original_query", "")

    true_chunks, noisy_chunks = hybrid_retriever.retrieve(
        query, source_filter=state.get("source_filter")
    )

    # Serialize chunks to dicts for state storage
    true_data_dicts = [
        {
            "content": chunk.document.page_content,
            "source": chunk.document.metadata.get("source_file", "unknown"),
            "source_type": chunk.document.metadata.get("source_type", "document"),
            "score": round(chunk.score, 4),
            "is_true_data": True,
            "chunk_index": chunk.document.metadata.get("chunk_index", 0),
        }
        for chunk in true_chunks
    ]

    noisy_data_dicts = [
        {
            "content": chunk.document.page_content[:200] + "...",  # truncate for state efficiency
            "source": chunk.document.metadata.get("source_file", "unknown"),
            "source_type": chunk.document.metadata.get("source_type", "document"),
            "score": round(chunk.score, 4),
            "is_true_data": False,
        }
        for chunk in noisy_chunks
    ]

    source_docs = list({c["source"] for c in true_data_dicts})

    metadata = {
        "true_data_count": len(true_chunks),
        "noisy_data_count": len(noisy_chunks),
        "avg_true_score": round(
            sum(c["score"] for c in true_data_dicts) / max(len(true_data_dicts), 1), 3
        ),
        "query_used": query,
    }

    steps = state.get("pipeline_steps", [])
    steps.append("retrieval")

    logger.info(
        "retrieval_complete",
        true_data=len(true_chunks),
        noisy_data=len(noisy_chunks),
    )

    return {
        "true_data_chunks": true_data_dicts,
        "noisy_data_chunks": noisy_data_dicts,
        "source_documents": source_docs,
        "retrieval_metadata": metadata,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 5: Context Assembly
# Format retrieved chunks into LLM prompt context
# ============================================================
def context_assembly_node(state: RAGState) -> Dict[str, Any]:
    """
    Format the True Data chunks into a structured context for the LLM prompt.

    Learning Note:
        HOW you format context matters.
        Including chunk source and score helps the LLM understand
        the relative importance of each piece of information.
        Numbered chunks make it easier to cite sources.
    """
    logger.info("node_context_assembly_start")

    true_chunks = state.get("true_data_chunks", [])

    if not true_chunks:
        context = "No relevant information found in the knowledge base."
    else:
        context = format_context(true_chunks)

    steps = state.get("pipeline_steps", [])
    steps.append("context_assembly")

    return {
        "assembled_context": context,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 6: LLM Generation
# Generate answer from context using the LLM via gateway
# ============================================================
def generation_node(state: RAGState) -> Dict[str, Any]:
    """
    Generate the final answer using the LLM.

    Learning Note:
        The SYSTEM PROMPT is crucial for RAG quality. We instruct the LLM:
        1. Answer ONLY from the provided context
        2. If context is insufficient, say so honestly
        3. Cite the source chunks
        4. Be concise and accurate

        This "grounding instruction" is the key to preventing hallucination.
        Without it, the LLM uses its parametric knowledge too freely.
    """
    logger.info("node_generation_start")
    from src.gateway.llm_gateway import llm_gateway

    query = state.get("rewritten_query") or state.get("original_query", "")
    context = state.get("assembled_context", "No context available.")
    history = state.get("conversation_history", [])

    # Build the prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]

    # Add recent history for conversational continuity. The UI sends the
    # current question as the last history item; don't repeat it.
    if history and history[-1].get("content") == state.get("original_query"):
        history = history[:-1]
    for msg in history[-4:]:  # last 2 turns
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    try:
        response = llm_gateway.complete(messages, temperature=0.1, max_tokens=1500)
        model_used = llm_gateway._build_model_string()
    except Exception as exc:
        err_str = str(exc)
        logger.warning("generation_failed_using_context_fallback", error=err_str[:100])

        # FALLBACK: Return retrieved context directly when no LLM is available.
        # Learning Note: This "retrieval-only" mode shows the RAG pipeline working
        # end-to-end even without an LLM. The retrieved chunks ARE the answer.
        # FAST FALLBACK: Use keyword-extraction to answer from context instantly
        context_text = state.get("assembled_context", "")
        true_chunks = state.get("true_data_chunks", [])
        if true_chunks or context_text:
            from src.gateway.simple_llm import extract_answer_from_context
            response = extract_answer_from_context(query, context_text)
            model_used = "keyword-extractor (fast-local)"
            logger.info("using_fast_local_fallback")
        else:
            response = "No documents found. Please upload a document first, then ask a question about it."
            model_used = "no-context"

    steps = state.get("pipeline_steps", [])
    steps.append("generation")

    logger.info("generation_complete", response_len=len(response))

    return {
        "llm_response": response,
        "model_used": model_used,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 7: Output Guardrails
# Validate LLM response before sending to user
# ============================================================
def output_guardrails_node(state: RAGState) -> Dict[str, Any]:
    """
    Validate the LLM response for safety, hallucination, and PII.
    """
    logger.info("node_output_guardrails_start")
    from src.guardrails.output_guardrails import output_guardrails

    response = state.get("llm_response", "")
    context_texts = [chunk["content"] for chunk in state.get("true_data_chunks", [])]

    result = output_guardrails.validate(response, context_texts)

    steps = state.get("pipeline_steps", [])
    steps.append("output_guardrails")

    return {
        "output_is_safe": result.is_safe,
        "output_violations": result.violations,
        "final_response": result.sanitized_response,
        "hallucination_score": result.hallucination_score,
        "pipeline_steps": steps,
    }


# ============================================================
# NODE 8: Evaluation
# Compute RAGAS quality metrics for this response
# ============================================================
def evaluation_node(state: RAGState) -> Dict[str, Any]:
    """
    Run inline RAGAS evaluation to measure response quality.

    Learning Note:
        This runs AFTER the response is generated, not before.
        The scores are returned to the UI so the user can see
        how confident the system is in its answer.

        In production, these scores are also logged to BigQuery
        for tracking quality trends over time.
    """
    logger.info("node_evaluation_start")
    from src.evals.evaluator import rag_evaluator

    question = state.get("original_query", "")
    answer = state.get("final_response", "")
    contexts = [chunk["content"] for chunk in state.get("true_data_chunks", [])]

    metrics = rag_evaluator.evaluate(question=question, answer=answer, contexts=contexts)

    steps = state.get("pipeline_steps", [])
    steps.append("evaluation")

    return {
        "eval_metrics": metrics.to_dict(),
        "pipeline_steps": steps,
    }


# ============================================================
# ERROR NODE: Return safe error response
# ============================================================
def error_response_node(state: RAGState) -> Dict[str, Any]:
    """
    Return a safe error message when input guardrails block a request.
    """
    violations = state.get("input_violations", [])
    steps = state.get("pipeline_steps", [])
    steps.append("error_response")

    error_msg = "Your request could not be processed due to security policy violations."
    if violations:
        # Give a helpful hint without revealing attack detection details
        if any("empty" in v.lower() for v in violations):
            error_msg = "Please provide a non-empty question."
        elif any("long" in v.lower() for v in violations):
            error_msg = "Your question is too long. Please shorten it and try again."

    return {
        "final_response": error_msg,
        "output_is_safe": False,
        "eval_metrics": {},
        "pipeline_steps": steps,
        "error": "; ".join(violations),
    }


# ============================================================
# CONDITIONAL EDGE FUNCTIONS
# These determine which node to route to next
# ============================================================

def route_after_input_guardrails(state: RAGState) -> str:
    """
    Route: if input is safe -> continue pipeline, else -> error.

    Learning Note:
        Conditional edges in LangGraph are just Python functions
        that return a string matching one of the defined edge names.
        This is how you implement if/else branching in the graph.
    """
    if state.get("input_is_safe", True):
        return "continue"
    return "blocked"
