# ============================================================
# src/api/routes/query.py - Query / Chat Endpoints
#
# Learning Note:
#   This is the MAIN endpoint users interact with.
#   POST /api/v1/query -> runs the full LangGraph RAG pipeline
#   POST /api/v1/query/stream -> streaming response (token by token)
#
#   Pydantic models define the request/response schema.
#   FastAPI auto-generates documentation from these models.
# ============================================================

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import json

from src.observability.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# Pydantic Request/Response Models
# Learning Note: These define the JSON structure for the API.
#   FastAPI validates incoming JSON against these models and
#   returns 422 Unprocessable Entity if validation fails.
# ============================================================

class ChatMessage(BaseModel):
    """A single message in conversation history."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class QueryRequest(BaseModel):
    """Request body for the query endpoint."""
    question: str = Field(..., min_length=1, max_length=4000, description="User's question")
    session_id: str = Field(default="", description="Session ID for conversation continuity")
    conversation_history: List[ChatMessage] = Field(
        default=[], description="Prior conversation messages for context"
    )
    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Registered source IDs to search. Omit both source fields to search everything.",
    )
    legacy_files: Optional[List[str]] = Field(
        default=None, description="Unregistered (pre-registry) uploaded files to search, by name",
    )

    def source_filter(self):
        from src.retrieval.source_filter import SourceFilter
        return SourceFilter.from_request(self.source_ids, self.legacy_files)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the main benefits of transformer models?",
                "session_id": "session-abc123",
                "conversation_history": []
            }
        }


class ChunkInfo(BaseModel):
    """Information about a retrieved document chunk."""
    content: str
    source: str
    score: float
    is_true_data: bool


class EvalScores(BaseModel):
    """RAG evaluation scores."""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    overall_score: float = 0.0
    evaluated: bool = False


class QueryResponse(BaseModel):
    """Response from the RAG pipeline."""
    answer: str
    sources: List[str]
    true_data_chunks: List[Dict[str, Any]]
    noisy_data_chunks: List[Dict[str, Any]]
    eval_metrics: Dict[str, Any]
    pipeline_steps: List[str]
    query_intent: str
    rewritten_query: str
    hallucination_score: float
    model_used: str
    latency_ms: int
    session_id: str
    input_safe: bool
    output_safe: bool
    error: Optional[str] = None


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_endpoint(
    request: QueryRequest,
    x_user_id: str = Header(default="anonymous", alias="X-User-ID"),
):
    """
    Main RAG query endpoint. Runs the full LangGraph pipeline:
    1. Input guardrails
    2. Query planning
    3. History-aware rewriting
    4. Hybrid retrieval (vector + BM25 + re-ranking)
    5. True Data vs Noisy Data classification
    6. LLM generation via gateway
    7. Output guardrails
    8. RAGAS evaluation

    Returns the answer along with retrieved chunks, eval scores,
    and pipeline execution trace.
    """
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        "query_received",
        session_id=session_id,
        user_id=x_user_id,
        question_len=len(request.question),
    )

    try:
        from src.graph.pipeline import rag_pipeline

        # Convert Pydantic models to dicts for the pipeline
        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]

        # Run the LangGraph pipeline
        result = rag_pipeline.run(
            query=request.question,
            session_id=session_id,
            user_id=x_user_id,
            conversation_history=history,
            source_filter=request.source_filter(),
        )

        result["session_id"] = session_id
        return QueryResponse(**result)

    except Exception as exc:
        logger.error("query_endpoint_error", error=str(exc), session_id=session_id)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")


@router.post("/query/stream", summary="Ask a question with streaming response")
async def query_stream_endpoint(
    request: QueryRequest,
    x_user_id: str = Header(default="anonymous", alias="X-User-ID"),
):
    """
    Streaming RAG endpoint. Returns tokens as Server-Sent Events (SSE).

    Learning Note:
        SSE (Server-Sent Events) is a simple protocol for pushing
        data from server to browser. The client connects once and
        receives a stream of "data: ..." lines.

        Format:
            data: {"token": "The", "done": false}\n\n
            data: {"token": " answer", "done": false}\n\n
            data: {"type": "metadata", "sources": [...], "done": true}\n\n
    """
    session_id = request.session_id or str(uuid.uuid4())
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    async def event_generator():
        try:
            from src.graph.pipeline import rag_pipeline
            from src.guardrails.input_guardrails import input_guardrails

            # Run guardrails first
            guard_result = input_guardrails.validate(request.question)
            if not guard_result.is_safe:
                error_event = json.dumps({"error": "Request blocked by guardrails", "done": True})
                yield f"data: {error_event}\n\n"
                return

            # Retrieve context, then stream generation
            from starlette.concurrency import run_in_threadpool
            from src.retrieval.hybrid_retriever import hybrid_retriever
            from src.graph.nodes import (build_messages, format_context, knowledge_catalog,
                                         no_context_answer, unindexed_sources, commit_list_answer)
            from src.gateway.llm_gateway import llm_gateway

            true_chunks, noisy_chunks = await run_in_threadpool(
                hybrid_retriever.retrieve,
                guard_result.sanitized_text,
                source_filter=request.source_filter(),
            )

            source_filter = request.source_filter()
            catalog = await run_in_threadpool(knowledge_catalog, source_filter)

            chunk_dicts = [
                {
                    "content": c.document.page_content,
                    "source": c.document.metadata.get("source_file", "unknown"),
                    "source_type": c.document.metadata.get("source_type", "document"),
                    "score": c.score,
                }
                for c in true_chunks
            ]
            direct = commit_list_answer(request.question, chunk_dicts)

            if not true_chunks:
                # Nothing relevant retrieved: answer honestly instead of letting the
                # model answer from general knowledge with invented citations.
                answer = no_context_answer(catalog, request.question, unindexed_sources(source_filter))
                yield f"data: {json.dumps({'token': answer, 'done': False})}\n\n"
            elif direct:
                # "Last N commits": exact list from the indexed history, no model needed.
                yield f"data: {json.dumps({'token': direct, 'done': False})}\n\n"
            else:
                messages = build_messages(guard_result.sanitized_text, format_context(chunk_dicts), catalog,
                                          history, request.question)
                async for token in llm_gateway.astream(messages, temperature=0.1, max_tokens=1500):
                    event = json.dumps({"token": token, "done": False})
                    yield f"data: {event}\n\n"

            # Send metadata after streaming completes
            sources = list(dict.fromkeys(
                c.document.metadata.get("source_file", "unknown") for c in true_chunks
            ))
            metadata = json.dumps({
                "done": True,
                "type": "metadata",
                "sources": sources,
                "true_data_count": len(true_chunks),
                "noisy_data_count": len(noisy_chunks),
                "session_id": session_id,
            })
            yield f"data: {metadata}\n\n"

        except Exception as exc:
            logger.error("stream_error", error=str(exc))
            error_event = json.dumps({"error": str(exc), "done": True})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
