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

from fastapi import APIRouter, HTTPException, Header, Request
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
    images: Optional[List[str]] = Field(
        default=None, description="Up to 3 images as data URLs (PNG, JPEG, WebP, GIF), answered by the vision model",
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


async def image_answer_events(request: "QueryRequest", history: list):
    """
    Steps for a question with images, as events: {"status"}, {"token"}, then {"sources", "model"}.
    1. the vision model reads the image once (short transcription);
    2. text from the image is searched exactly in the sources, plus normal search;
    3. "where is the code?" is answered from the matches directly; other questions
       are answered by the faster text model using what the image shows.
    """
    from starlette.concurrency import run_in_threadpool
    from src.gateway.llm_gateway import llm_gateway
    from src.graph.image_flow import (ImageModelError, build_answer_messages, extract_search_terms,
                                      find_code_matches, format_locate_answer, is_locate_question, read_image)
    from src.graph.nodes import decode_images, format_context, knowledge_catalog
    from src.retrieval.hybrid_retriever import hybrid_retriever

    images = decode_images(request.images or [])          # ValueError for bad images
    vision_model = llm_gateway.vision_model_string()
    yield {"status": "Reading the image... (on a PC without a GPU this can take a minute)"}
    try:
        transcription = await run_in_threadpool(read_image, images, vision_model)
    except ImageModelError as exc:
        yield {"token": str(exc)}
        yield {"sources": [], "model": "none (image model unavailable)"}
        return
    except Exception as exc:
        yield {"token": _vision_error(exc)}
        yield {"sources": [], "model": "none (image model unavailable)"}
        return

    yield {"status": "Searching your sources for text from the image..."}
    source_filter = request.source_filter()
    terms = extract_search_terms(transcription)
    matches = await run_in_threadpool(find_code_matches, terms, source_filter)
    # One line: a multi-line query would be split into separate sub-questions.
    query = " ".join([request.question] + terms)
    true_chunks, _ = await run_in_threadpool(hybrid_retriever.retrieve, query, source_filter=source_filter)
    chunk_files = list(dict.fromkeys(c.document.metadata.get("source_file", "unknown") for c in true_chunks))
    sources = list(dict.fromkeys([m["file"] for m in matches] + chunk_files))
    logger.info("image_question", terms=terms[:8], code_matches=len(matches), chunks=len(true_chunks))

    if is_locate_question(request.question):
        yield {"token": format_locate_answer(matches, chunk_files, transcription)}
        yield {"sources": sources, "model": f"{vision_model} (read) + exact search"}
        return

    yield {"status": "Writing the answer..."}
    context = format_context([{
        "content": c.document.page_content,
        "source": c.document.metadata.get("source_file", "unknown"),
        "source_type": c.document.metadata.get("source_type", "document"),
        "score": c.score,
    } for c in true_chunks])
    catalog = await run_in_threadpool(knowledge_catalog, source_filter)
    messages = build_answer_messages(request.question, transcription, matches, context, catalog, history)
    async for token in llm_gateway.astream(messages, temperature=0.1, max_tokens=1200):
        yield {"token": token}
    yield {"sources": sources, "model": f"{vision_model} (read) + {llm_gateway._build_model_string()}"}


def _vision_error(exc: Exception) -> str:
    text = str(exc)
    if "not found" in text.lower() and ("model" in text.lower() or "pull" in text.lower()):
        from src.gateway.llm_gateway import llm_gateway
        model = llm_gateway.vision_model_string().split("/", 1)[1]
        return (f"The image model **{model}** is not installed in Ollama. Run `ollama pull {model}` once, "
                "then ask again.")
    return f"Could not analyse the image: {text[:300]}"


@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_endpoint(
    request: QueryRequest,
    http: Request,
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
    x_user_id = http.state.user.username

    logger.info(
        "query_received",
        session_id=session_id,
        user_id=x_user_id,
        question_len=len(request.question),
    )

    history = [{"role": msg.role, "content": msg.content} for msg in request.conversation_history]
    if request.images:
        import time
        started = time.time()
        answer, sources, model = [], [], "unknown"
        try:
            async for event in image_answer_events(request, history):
                answer.append(event.get("token", ""))
                sources, model = event.get("sources", sources), event.get("model", model)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return QueryResponse(answer="".join(answer), sources=sources, true_data_chunks=[], noisy_data_chunks=[],
                             eval_metrics={}, pipeline_steps=["read image", "exact search", "retrieval", "answer"],
                             query_intent="image", rewritten_query=request.question, hallucination_score=0.0,
                             model_used=model, latency_ms=int((time.time() - started) * 1000),
                             session_id=session_id, input_safe=True, output_safe=True)

    try:
        from src.graph.pipeline import rag_pipeline

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


class SuggestionsRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    answer: str = Field(default="", max_length=20000)
    sources: List[str] = Field(default_factory=list)
    source_ids: Optional[List[str]] = None
    legacy_files: Optional[List[str]] = None


@router.post("/query/suggestions", summary="Suggest follow-up questions for an answer")
async def suggestions_endpoint(request: SuggestionsRequest):
    """Three follow-up questions grounded in the answer's sources (starter questions if it had none)."""
    from starlette.concurrency import run_in_threadpool
    from src.graph.nodes import knowledge_catalog, suggest_followups
    from src.retrieval.source_filter import SourceFilter

    source_filter = SourceFilter.from_request(request.source_ids, request.legacy_files)
    catalog = await run_in_threadpool(knowledge_catalog, source_filter)
    suggestions = await run_in_threadpool(suggest_followups, request.question, request.answer,
                                          request.sources, catalog)
    return {"suggestions": suggestions}


@router.post("/query/stream", summary="Ask a question with streaming response")
async def query_stream_endpoint(
    request: QueryRequest,
    http: Request,
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
    x_user_id = http.state.user.username
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
                                         no_context_answer, unindexed_sources, commit_list_answer,
                                         missing_commit_history, is_code_request, build_code_messages,
                                         CODE_MODEL_HINT)
            from src.gateway.llm_gateway import llm_gateway

            if request.images:
                # Image questions: read once, exact search, then answer (status updates while slow steps run).
                try:
                    async for event in image_answer_events(request, history):
                        if "sources" in event:
                            yield f"data: {json.dumps({'done': True, 'type': 'metadata', 'sources': event['sources'], 'model': event['model'], 'true_data_count': len(event['sources']), 'noisy_data_count': 0, 'session_id': session_id})}\n\n"
                        else:
                            yield f"data: {json.dumps({**event, 'done': False})}\n\n"
                except ValueError as exc:
                    yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
                return

            # Jira filter questions (who / when / status) are answered by a live JQL search.
            from src.sources.jira_query import answer_jira_question
            jira = await run_in_threadpool(answer_jira_question, request.question, request.source_filter())
            if jira:
                answer, keys = jira
                yield f"data: {json.dumps({'token': answer, 'done': False})}\n\n"
                yield f"data: {json.dumps({'done': True, 'type': 'metadata', 'sources': keys, 'true_data_count': 0, 'noisy_data_count': 0, 'session_id': session_id})}\n\n"
                return

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
                    "metadata": {k: c.document.metadata.get(k) for k in ("source_id", "file_path", "chunk_index")},
                }
                for c in true_chunks
            ]
            direct = commit_list_answer(request.question, chunk_dicts)

            if is_code_request(request.question) and not direct:
                # Code suggestion mode: code model + expanded file context; falls back to the
                # general model (with an install hint) if the code model is not installed.
                messages = await run_in_threadpool(build_code_messages, request.question, chunk_dicts, catalog,
                                                   history)
                code_model = llm_gateway.code_model_string()
                yield f"data: {json.dumps({'status': 'Writing code with ' + code_model.split('/', 1)[1] + '...', 'done': False})}\n\n"
                started = False
                try:
                    async for token in llm_gateway.astream(messages, temperature=0.2, max_tokens=2000,
                                                           model=code_model):
                        started = True
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                except Exception as exc:
                    if started:
                        raise
                    logger.warning("code_model_unavailable_using_default", error=str(exc)[:200])
                    async for token in llm_gateway.astream(messages, temperature=0.2, max_tokens=2000):
                        yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                    yield f"data: {json.dumps({'token': CODE_MODEL_HINT, 'done': False})}\n\n"
            elif not true_chunks:
                # Nothing relevant retrieved: answer honestly instead of letting the
                # model answer from general knowledge with invented citations.
                answer = no_context_answer(catalog, request.question, unindexed_sources(source_filter),
                                           missing_commit_history(request.question, source_filter))
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
