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

import re
import time
from typing import Dict, Any, List, Optional, Tuple

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


SOURCE_TYPE_LABELS = {"document": "document", "bitbucket": "code", "jira": "Jira issue",
                      "confluence": "Confluence page"}

# Shared by the graph's generation node and the streaming endpoint.
SYSTEM_PROMPT = """You are a precise engineering assistant. Answer using ONLY the numbered
sources below. They come from the user's uploaded documents, Bitbucket repository
files, Jira issues, and Confluence pages.

How to answer:
- If the user asks several questions, answer each one under its own short heading.
- Explain in your own words, then cite in parentheses the file path or Jira key
  copied exactly from the header of the numbered source you used.
- Only cite paths and keys that appear in the source headers below. Never make one up.
- For code, name the files, functions, and steps involved; quote a few key lines when useful.
- If the sources do not cover a question, say so for that question only and answer the rest.
- If the user asks what you know or how you can help, describe the knowledge sources listed
  below instead of general abilities.
- Never invent files, functions, or behaviour that the sources do not show.

Knowledge sources the user selected:
{catalog}

Sources:
{context}"""


def knowledge_catalog(source_filter=None) -> List[str]:
    """Readable list of the searchable sources, limited to the chat selection."""
    try:
        from src.sources.registry import source_registry
        from src.retrieval.vector_store import vector_store
        kinds = {"document": "Document", "bitbucket": "Bitbucket repository", "jira": "Jira project",
                 "confluence": "Confluence space"}
        lines = []
        for src in source_registry.list_sources():
            if not src.chunk_count:
                continue
            if source_filter is not None and src.id not in source_filter.source_ids:
                continue
            lines.append(f"- {kinds.get(src.type.value, src.type.value)}: {src.name} ({src.chunk_count} chunks)")
        for legacy in vector_store.list_legacy_files():
            if source_filter is None or legacy["source_file"] in source_filter.legacy_files:
                lines.append(f"- Older upload: {legacy['source_file']} ({legacy['chunk_count']} chunks)")
        return lines
    except Exception as exc:
        logger.warning("knowledge_catalog_failed", error=str(exc))
        return []


_ABOUT_ASSISTANT = re.compile(
    r"\b(help me|can you (do|help)|what (can|do) you|your (knowledge|sources)|"
    r"(what'?s?|which) (knowledge|sources|data|documents)|knowledge (do )?you have|you know|"
    r"about yourself|who are you)\b", re.I)


def unindexed_sources(source_filter=None) -> List[str]:
    """Selected repositories / projects / documents that have no searchable chunks yet."""
    try:
        from src.sources.registry import source_registry
        return [src.name for src in source_registry.list_sources()
                if not src.chunk_count and (source_filter is None or src.id in source_filter.source_ids)]
    except Exception:
        return []


def missing_commit_history(query: str, source_filter=None) -> List[str]:
    """Selected, indexed repositories that have no commit history yet (for commit questions)."""
    from src.retrieval.hybrid_retriever import COMMIT_INTENT
    if not COMMIT_INTENT.search(query or ""):
        return []
    try:
        from src.sources.registry import source_registry
        from src.retrieval.vector_store import vector_store
        return [src.name for src in source_registry.list_sources()
                if src.type.value == "bitbucket" and src.chunk_count
                and (source_filter is None or src.id in source_filter.source_ids)
                and not vector_store.has_commit_history(src.id)]
    except Exception:
        return []


def no_context_answer(catalog: List[str], query: str = "", pending: List[str] = None,
                      no_history: List[str] = None) -> str:
    """Answer used instead of the LLM when retrieval found nothing, so nothing is invented."""
    if no_history:
        return ("Commit history hasn't been indexed yet for: " + ", ".join(no_history) + ".\n\n"
                "It was indexed before commit history was supported. Open **Knowledge Sources > Bitbucket** "
                "and click **Reindex** (or **Sync**) on the repository, wait for the green READY badge, "
                "then ask again.")
    note = ""
    if pending:
        note = ("\n\n**Not searchable yet:** " + ", ".join(pending) + ". Open **Knowledge Sources**, "
                "click **Index** on it, and wait for the green READY badge.")
    return _no_context_answer(catalog, query) + note


_GREETING = re.compile(r"^\s*(hi+|hello|hey|hii+|good (morning|afternoon|evening)|namaste|thanks|thank you)\b"
                       r"[\s!.,?]*$", re.I)


def _no_context_answer(catalog: List[str], query: str = "") -> str:
    if _GREETING.match(query or ""):
        if not catalog:
            return ("Hi! I answer questions from your own knowledge sources, but none are searchable yet. "
                    "Add documents, a Bitbucket repository, or a Jira project under **Knowledge Sources**.")
        return ("Hi! I answer questions using only your connected knowledge sources and cite where "
                "each answer comes from.\n\n**Sources I can search right now:**\n" + "\n".join(catalog) +
                "\n\nFor example, ask how a feature works, which files implement something, "
                "or what the latest commits were.")
    if catalog and _ABOUT_ASSISTANT.search(query or ""):
        return ("I answer questions using only the knowledge sources you've connected, and I cite "
                "the file, document, or Jira issue each answer comes from.\n\n"
                "**Sources I can search right now:**\n" + "\n".join(catalog) +
                "\n\nAsk about something in them, for example how a feature works, which files "
                "implement something, or what an issue is about.")
    if not catalog:
        return ("Your knowledge base has nothing to search yet (or no sources are selected).\n\n"
                "Add documents, a Bitbucket repository, or a Jira project under **Knowledge Sources** "
                "(click **Index** for repositories and projects), tick them under **Search In**, "
                "then ask again.")
    return ("I couldn't find anything relevant to that in the selected sources, so I won't guess.\n\n"
            "**I can answer questions about:**\n" + "\n".join(catalog) +
            "\n\nTry asking about something specific in these sources, for example a feature, "
            "file, workflow, or issue, or select more sources under **Search In**.")


_COMMIT_LIST = re.compile(
    r"\b(?:last|latest|recent|newest|top)\s+(\d{1,3})\s+commits?\b"
    r"|\b(?:list|show|give|tell|display)\b[^?]*\b(?:last|latest|recent|newest)\s+commits\b", re.I)
_COMMIT_LINE = re.compile(r"^(\d+)\. (\S+) \| ([^|]*) \| ([^|]*) \| (.*)$")


def commit_list_answer(query: str, chunks: List[Dict]) -> str:
    """
    Answer "last N commits" directly from the indexed commit history, so the list is
    exact and in order (a small model re-orders and skips lines). None if not applicable.
    """
    match = _COMMIT_LIST.search(query or "")
    if not match:
        return None
    count = min(int(match.group(1)), 50) if match.group(1) else 10
    repos = {}
    for chunk in chunks:
        if not chunk.get("source", "").endswith("/(commit history)"):
            continue
        repo = chunk["source"][: -len("/(commit history)")]
        for line in chunk.get("content", "").splitlines():
            parsed = _COMMIT_LINE.match(line.strip())
            if parsed:
                repos.setdefault(repo, {})[int(parsed.group(1))] = parsed.groups()[1:]
    if not repos:
        return None
    parts = []
    for repo, commits in repos.items():
        rows = [commits[n] for n in sorted(commits)[:count]]
        parts.append(f"### Last {len(rows)} commits in {repo}\n\n| # | Commit | Date | Author | Message |\n"
                     "|---|---|---|---|---|\n" +
                     "\n".join(f"| {i} | `{h}` | {d.strip()} | {a.strip()} | {m.strip().replace('|', '/')} |"
                                for i, (h, d, a, m) in enumerate(rows, 1)))
    return "\n\n".join(parts) + "\n\n(from the indexed commit history, newest first)"


def default_suggestions(catalog: List[str]) -> List[str]:
    """Starter questions that work for whatever kinds of sources are indexed."""
    text = "\n".join(catalog)
    suggestions = []
    if "Bitbucket repository" in text:
        suggestions += ["Explain this project", "What are the latest 5 commits?",
                        "Which files handle configuration?"]
    if "Jira project" in text:
        suggestions.append("What are the most recent Jira issues?")
    if "Confluence space" in text:
        suggestions.append("Summarize the main Confluence pages")
    if "Document:" in text or "Older upload:" in text:
        suggestions.append("Summarize the uploaded documents")
    return suggestions[:3]


def _clean_suggestion(line: str) -> str:
    line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip().strip('"').strip()
    return line if 8 <= len(line) <= 120 and line.endswith("?") else ""


def suggest_followups(question: str, answer: str, sources: List[str], catalog: List[str]) -> List[str]:
    """Three follow-up questions grounded in the answer and its source files."""
    if not sources:
        return default_suggestions(catalog)
    from src.gateway.llm_gateway import llm_gateway
    files = "\n".join(f"- {s}" for s in sources[:8])
    messages = [
        {"role": "system", "content": (
            "Suggest exactly 3 short follow-up questions a developer could ask next. Each must be about "
            "something mentioned in the answer or in the listed source files, be answerable from those "
            "sources, and differ from the original question. Output one question per line, each ending "
            "with '?', with no numbering and no other text.")},
        {"role": "user", "content": f"Question: {question}\n\nAnswer:\n{answer[:3000]}\n\nSource files:\n{files}"},
    ]
    try:
        raw = llm_gateway.complete(messages, temperature=0.3, max_tokens=150)
    except Exception as exc:
        logger.warning("suggestions_failed", error=str(exc)[:200])
        return default_suggestions(catalog)
    seen, out = {question.strip().lower()}, []
    for line in raw.splitlines():
        cleaned = _clean_suggestion(line)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            out.append(cleaned)
    return out[:3] or default_suggestions(catalog)


MAX_IMAGES = 3
MAX_IMAGE_BYTES = 6 * 1024 * 1024
_DATA_URL = re.compile(r"^data:image/(png|jpe?g|webp|gif);base64,([A-Za-z0-9+/=\s]+)$", re.I)


def decode_images(images: List[str]) -> List[str]:
    """Validate data-URL images from the browser; returns the raw base64 strings Ollama expects."""
    import base64
    if len(images) > MAX_IMAGES:
        raise ValueError(f"Attach at most {MAX_IMAGES} images")
    out = []
    for image in images:
        match = _DATA_URL.match(image or "")
        if not match:
            raise ValueError("Images must be PNG, JPEG, WebP or GIF")
        data = "".join(match.group(2).split())
        try:
            size = len(base64.b64decode(data, validate=True))
        except ValueError as exc:
            raise ValueError("An attached image is not valid") from exc
        if size > MAX_IMAGE_BYTES:
            raise ValueError("Each image must be under 6 MB")
        out.append(data)
    return out


# "write / implement / fix / refactor / add tests ..." -> code suggestion mode.
CODE_INTENT = re.compile(
    r"\b(?:suggest|give me|show me|write|generate|create|implement|add|fix|refactor|rewrite|modify|change|update|"
    r"optimi[sz]e|convert|extend)\b.{0,60}?\b(?:code|function|method|class|tests?|unit tests?|endpoint|api|"
    r"component|script|query|handler|validation|logic|feature|retry|hook|service|controller|migration|snippet|"
    r"implementation)\b"
    r"|\bhow (?:do|would|can|should) i (?:implement|write|code|add|build|fix)\b"
    r"|\b(?:code|snippet|example) (?:for|to|that)\b", re.I | re.S)

CODE_PROMPT = """You are a senior software engineer pairing with the user on THEIR codebase. Write the code
they ask for so it fits this repository.

How to answer:
- Base the code on the repository context below: reuse its existing functions, types, libraries,
  naming, error handling and test style. Name the file(s) to change and where (function / class).
- Show code in fenced blocks with the language, e.g. ```ts. For a change to an existing function,
  show the whole updated function; for new code, show where it goes.
- Briefly explain the change and list any assumptions. Add a test when it is natural.
- If the context does not show something you need (a type, a helper, a config value), say so and
  mark that part of the code as an assumption instead of presenting it as existing code.
- This is a suggestion: never claim it has been applied or committed.

Knowledge sources the user selected:
{catalog}

Repository context (numbered):
{context}"""


CODE_MODEL_HINT = ("\n\n_Written with the general model because the code model is not installed. For better "
                   "code run `ollama pull qwen2.5-coder:7b` (or set OLLAMA_CODE_MODEL)._")


FILE_EXPLAIN_PROMPT = """You are a senior engineer explaining a file from the user's repository.
The user can already see the code, so NEVER copy code blocks back: do not use ``` fences, and quote
at most one signature line per block in a single `code span`.
Explain it block by block, in the order the code appears:
- For each block (imports, constants, types, each function / class / component, config sections),
  start with a short heading naming it, give its signature in a code span, then explain in 2-5
  sentences or bullets what it does, its inputs and outputs, and how it connects to the rest.
- Point out anything important: side effects, error handling, API endpoints, magic values, TODOs.
- End with a 2-3 line summary of the file's role in the project.
- Explain only what is in the file; do not invent code. If the file was cut short, say so.

File: {path}
```
{content}
```"""

FILE_EXPLAIN_INTENT = re.compile(
    r"\b(explain|explanation|walk me through|describe|summari[sz]e|review|understand|line by line|"
    r"each (?:code )?block|block by block|in detail|more detail|what does (?:it|this|the file|this file) do)\b", re.I)
_FILE_TOKEN = re.compile(r"[\w@.\[\]-]+(?:/[\w@.\[\]-]+)*\.[A-Za-z0-9]{1,6}\b|[\w-]+(?:/[\w@.\[\]-]+)+")
MAX_FILE_CHARS = 14000


def find_file_reference(text: str, source_filter=None) -> Optional[tuple]:
    """(source_id, source_file, file_path) for a file named in the text: a path or a unique file name."""
    from src.retrieval.vector_store import vector_store
    tokens = [t.strip("`'\"()") for t in _FILE_TOKEN.findall(text or "")]
    if not tokens:
        return None
    files = vector_store.list_files(None if source_filter is None else source_filter.source_ids)
    for token in sorted(tokens, key=len, reverse=True):
        token_l = token.lower().lstrip("./")
        exact = [f for f in files if f[2].lower() == token_l or f[1].lower() == token_l]
        if exact:
            return exact[0]
        suffix = [f for f in files if f[2].lower().endswith("/" + token_l)]
        if len(suffix) == 1 or (suffix and "/" in token_l):
            return sorted(suffix, key=lambda f: len(f[2]))[0]
    return None


REVIEW_INTENT = re.compile(
    r"\b(?:what (?:changes?|would you change|should (?:i|we) change|to change|can be improved|is wrong)|"
    r"changes? (?:here|to this|in this|you want|would you)|want(?:s)? changes?|suggest(?:ed)? (?:changes?|improvements?)|"
    r"review|improve(?:ments?)?|refactor(?:ing)?|clean ?up|code smells?|best practices?|"
    r"(?:issues?|problems?|bugs?|mistakes?) (?:in|with) (?:this|it|the (?:file|code))|optimi[sz]e)\b", re.I)
# A short follow-up that points at what was just discussed ("change here", "improve it").
_DEICTIC = re.compile(r"\b(?:here|this (?:file|code|one)|that (?:file|code)|it)\b", re.I)


_BACK_REFERENCE = re.compile(
    r"\b(?:this|that|it|here|above|same|these|those|again)\b|\beach (?:code )?block\b|"
    r"\bblock by block\b|\bline by line\b|\bmore detail|\bin detail\b|\bfurther\b", re.I)


def is_follow_up(question: str) -> bool:
    """
    Does the question point back at what was just discussed ("explain each block", "what
    would you change here", "more detail")? Not when it names a new subject: a code name,
    the whole project, or the architecture - those are fresh questions.
    """
    from src.retrieval.hybrid_retriever import code_names, is_overview_question
    q = question or ""
    if not _BACK_REFERENCE.search(q) or code_names(q) or is_overview_question(q) or ARCH_INTENT.search(q):
        return False
    return bool(FILE_EXPLAIN_INTENT.search(q) or REVIEW_INTENT.search(q) or CODE_INTENT.search(q)
                or len(q.split()) <= 8)


def is_review_request(question: str) -> bool:
    return bool(REVIEW_INTENT.search(question or ""))


def file_for_question(question: str, history: List[Dict], source_filter=None) -> Optional[tuple]:
    """
    The file a question is about: named in it, or - for a follow-up such as "explain each
    block", "what would you change here" or "refactor it" - named in one of the user's last
    few messages.
    """
    found = find_file_reference(question, source_filter)
    if found or not is_follow_up(question):
        return found
    for message in reversed([m for m in history[:-1] if m.get("role") == "user"][-2:]):
        found = find_file_reference(message.get("content", ""), source_filter)
        if found:
            return found
    return None


def whole_file(source_id: str, file_path: str) -> Tuple[str, bool]:
    """The file's text reassembled from its chunks (header and chunk overlaps removed); (text, truncated)."""
    from src.retrieval.vector_store import vector_store
    chunks = vector_store.get_file_chunks(source_id, file_path, 0, limit=10_000)
    text = ""
    for doc in chunks:
        body = doc.page_content.split("\n\n", 1)[1] if doc.page_content.startswith("Repository:") else doc.page_content
        # Chunks overlap by up to ~100 characters: skip the part already added.
        overlap = next((n for n in range(min(len(text), len(body), 150), 0, -1) if text.endswith(body[:n])), 0)
        text += body[overlap:] if text else body
    return text[:MAX_FILE_CHARS], len(text) > MAX_FILE_CHARS


def is_file_explain_request(question: str) -> bool:
    return bool(FILE_EXPLAIN_INTENT.search(question or "")) and not is_code_request(question) \
        and not is_review_request(question)


FILE_REVIEW_PROMPT = """You are a careful senior engineer reviewing a file from the user's repository.
Give at most 5 improvements, most important first, and only ones you are confident are real
improvements for THIS code. For each one:
- a short heading and the line it concerns, quoted in a `code span`,
- why it matters (bug risk, error handling, security, readability, duplication, performance),
- the change, with a short before/after snippet only if it makes the change clear.
Rules:
- Never suggest removing input validation, sanitising (such as trim) or error handling.
- Only suggest let -> const when the variable is never reassigned in the code shown.
- Do not repeat the same kind of suggestion; do not copy large parts of the file.
- If the code is already good, say so and list fewer items. Base everything on the code below.

File: {path}
```
{content}
```"""


def build_file_explain_messages(file_path: str, content: str, truncated: bool, question: str,
                                history: List[Dict], review: bool = False) -> List[Dict]:
    note = "\n(... file continues; only the first part is shown ...)" if truncated else ""
    prompt = FILE_REVIEW_PROMPT if review else FILE_EXPLAIN_PROMPT
    messages = [{"role": "system", "content": prompt.format(path=file_path, content=content + note)}]
    prior = history[:-1] if history and history[-1].get("content") == question else history
    messages += [{"role": m["role"], "content": m["content"]} for m in prior[-2:]]
    messages.append({"role": "user", "content": question})
    return messages


ARCH_INTENT = re.compile(
    r"\b(?:arch|archi|architecture|architectural|diagram|draw|flow ?chart|block diagram|component diagram|"
    r"system design|high[- ]level design|hld|how (?:is|are) (?:the|this) (?:project|repo|repository|system|code) "
    r"(?:structured|organi[sz]ed|laid out)|(?:project|repo|repository|code ?base|folder) structure)\b", re.I)

ARCH_PROMPT = """You are a software architect describing the user's repository from the material below
(the real folder layout and key files, plus README / design docs). Answer with:

1. **Overview** - 2-3 sentences: what the system is and its main parts.
2. **Components** - one bullet per major part, named after the REAL top-level folders in the layout
   (and what README / package.json say they are): what it does and its folder in parentheses.
3. **How they interact** - the main request / data flows in a few bullets.
4. **Diagram** - a Mermaid flowchart in a ```mermaid block. Rules so it renders:
   start with "flowchart LR"; node ids are letters/digits only; write every label in double quotes
   with the closing quote BEFORE the bracket, in the form N1["label text"]; edges in the form
   N1 -->|"label text"| N2; no other text inside the block.
   Draw ONE node per component (a top-level folder, service, database or external system such as an
   API or webhook provider) - not one node per file or route - and each node only once; 5 to 12
   nodes; arrows show who calls whom (users -> UI -> server/API -> data and external services).

Use only components, folders and technologies that the layout, README or package.json show. Do not
assume a framework (for example Angular, React, Express, Django) unless the material names it.

Repository layout:
{layout}

Documentation and key files:
{context}"""

_KEY_FILES = re.compile(r"(?:^|/)(?:package\.json|angular\.json|tsconfig\.json|requirements\.txt|pyproject\.toml|"
                        r"pom\.xml|build\.gradle|go\.mod|Dockerfile|docker-compose\.ya?ml|Jenkinsfile|"
                        r"(?:server|app|main|index)\.(?:js|ts|py)|\.env\.example|README\.md)$", re.I)


def is_architecture_request(question: str) -> bool:
    return bool(ARCH_INTENT.search(question or ""))


def repo_layout(source_filter=None, max_lines: int = 70) -> Tuple[str, List[str]]:
    """
    A compact, exact layout of the selected repositories from the index: top-level folders with
    file counts and their main sub-folders, plus key files. Returns (text, repository names).
    """
    from collections import Counter, defaultdict
    from src.retrieval.vector_store import vector_store
    files = vector_store.list_files(None if source_filter is None else source_filter.source_ids)
    by_repo = defaultdict(list)
    for source_id, source_file, path in files:
        repo = source_file[: -len(path) - 1] if source_file.endswith("/" + path) else source_id
        by_repo[repo].append(path)
    lines, repos = [], list(by_repo)[:3]
    for repo in repos:
        paths = by_repo[repo]
        lines.append(f"Repository {repo} ({len(paths)} files):")
        top = Counter(p.split("/")[0] if "/" in p else "(root files)" for p in paths)
        for folder, count in top.most_common(12):
            if folder == "(root files)":
                root = [p for p in paths if "/" not in p]
                lines.append(f"  (root) {', '.join(sorted(root)[:12])}")
                continue
            subs = Counter(p.split("/")[1] for p in paths if p.startswith(folder + "/") and p.count("/") >= 2)
            sub_text = ", ".join(f"{name}/ ({n})" for name, n in subs.most_common(8))
            lines.append(f"  {folder}/ ({count} files){': ' + sub_text if sub_text else ''}")
        key = [p for p in paths if _KEY_FILES.search(p)][:15]
        if key:
            lines.append("  key files: " + ", ".join(key))
    return "\n".join(lines[:max_lines]) or "(no indexed repository files)", repos


def build_architecture_messages(question: str, layout: str, chunks: List[Dict], history: List[Dict]) -> List[Dict]:
    context = format_context(chunks) or "(no documentation found)"
    messages = [{"role": "system", "content": ARCH_PROMPT.format(layout=layout, context=context[:9000])}]
    prior = history[:-1] if history and history[-1].get("content") == question else history
    messages += [{"role": m["role"], "content": m["content"]} for m in prior[-2:]]
    messages.append({"role": "user", "content": question})
    return messages


def architecture_inputs(question: str, history: List[Dict], source_filter=None) -> Tuple[List[Dict], List[str]]:
    """Messages for an architecture answer and the sources used (layout + README / design docs)."""
    from src.retrieval.vector_store import vector_store
    layout, _ = repo_layout(source_filter)
    docs = vector_store.get_overview_chunks(source_filter, per_source=5)
    chunks = [{"content": d.page_content, "source": d.metadata.get("source_file", ""),
               "source_type": d.metadata.get("source_type", "document"), "score": 1.0} for d in docs]
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    return build_architecture_messages(question, layout, chunks, history), sources


def is_code_request(question: str) -> bool:
    return bool(CODE_INTENT.search(question or ""))


def code_context_chunks(chunks: List[Dict], files: int = 3, per_file: int = 8) -> List[Dict]:
    """
    The retrieved chunks, with the best-matching code files expanded to consecutive
    chunks, so the model sees whole functions and the file's conventions.
    """
    from src.retrieval.vector_store import vector_store
    out, seen_files = [], []
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        if (chunk.get("source_type") == "bitbucket" and meta.get("file_path")
                and not meta["file_path"].startswith("(") and len(seen_files) < files
                and (meta.get("source_id"), meta["file_path"]) not in seen_files):
            seen_files.append((meta.get("source_id"), meta["file_path"]))
            try:
                for doc in vector_store.get_file_chunks(meta.get("source_id"), meta["file_path"],
                                                        meta.get("chunk_index", 0), per_file):
                    out.append({"content": doc.page_content, "source": doc.metadata.get("source_file", ""),
                                "source_type": "bitbucket", "score": chunk.get("score", 1.0)})
                continue
            except Exception as exc:
                logger.warning("code_context_expand_failed", error=str(exc)[:120])
        if not any(c["content"] == chunk.get("content") for c in out):
            out.append(chunk)
    return out


def build_code_messages(question: str, chunks: List[Dict], catalog: List[str], history: List[Dict]) -> List[Dict]:
    context = format_context(code_context_chunks(chunks)) or "(no matching repository code was found)"
    messages = [{"role": "system", "content": CODE_PROMPT.format(context=context,
                                                                 catalog="\n".join(catalog) or "(none)")}]
    if history and history[-1].get("content") == question:
        history = history[:-1]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-4:]]
    messages.append({"role": "user", "content": question})
    return messages


def build_messages(query: str, context: str, catalog: List[str], history: List[Dict],
                   original_query: str) -> List[Dict]:
    """System prompt + last 2 turns + question. The UI sends the current question as the last history item."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
        context=context, catalog="\n".join(catalog) or "(none)")}]
    if history and history[-1].get("content") == original_query:
        history = history[:-1]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-4:]]
    messages.append({"role": "user", "content": query})
    return messages


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
    if not true_chunks and is_follow_up(state.get("original_query", "")):
        # A follow-up often only makes sense together with the previous question.
        history = state.get("conversation_history", [])
        previous = next((m["content"] for m in reversed(history[:-1]) if m.get("role") == "user"), "")
        if previous:
            true_chunks, noisy_chunks = hybrid_retriever.retrieve(
                f"{previous} {query}", source_filter=state.get("source_filter"))

    # Serialize chunks to dicts for state storage
    true_data_dicts = [
        {
            "content": chunk.document.page_content,
            "source": chunk.document.metadata.get("source_file", "unknown"),
            "source_type": chunk.document.metadata.get("source_type", "document"),
            "score": round(chunk.score, 4),
            "is_true_data": True,
            "chunk_index": chunk.document.metadata.get("chunk_index", 0),
            "metadata": {k: chunk.document.metadata.get(k) for k in ("source_id", "file_path", "chunk_index")},
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
    catalog = knowledge_catalog(state.get("source_filter"))

    steps = state.get("pipeline_steps", [])
    # Jira filter questions (who / when / status) are answered by a live JQL search.
    from src.sources.jira_query import answer_jira_question
    jira = answer_jira_question(state.get("original_query", ""), state.get("source_filter"))
    if jira:
        steps.append("generation")
        return {"llm_response": jira[0], "model_used": "none (live Jira search)", "pipeline_steps": steps}

    question = state.get("original_query", "")
    file_ref = file_for_question(question, state.get("conversation_history", []), state.get("source_filter"))
    review = bool(file_ref) and is_review_request(question)
    arch = not file_ref and is_architecture_request(question)
    if (file_ref and (review or is_file_explain_request(question))) or arch:
        from src.gateway.llm_gateway import llm_gateway as gateway
        if arch:
            messages, arch_sources = architecture_inputs(question, state.get("conversation_history", []),
                                                         state.get("source_filter"))
        else:
            content, truncated = whole_file(file_ref[0], file_ref[2])
            messages = build_file_explain_messages(file_ref[2], content, truncated, question,
                                                   state.get("conversation_history", []), review=review)
        try:
            model = gateway.code_model_string()
            response = gateway.complete(messages, temperature=0.2, max_tokens=2500, model=model)
        except Exception as exc:
            logger.warning("code_model_unavailable_using_default", error=str(exc)[:200])
            model = gateway._build_model_string()
            response = gateway.complete(messages, temperature=0.2, max_tokens=2500) + CODE_MODEL_HINT
        steps.append("generation")
        return {"llm_response": response, "model_used": model, "pipeline_steps": steps,
                "source_documents": arch_sources if arch else [file_ref[1]]}

    if is_code_request(state.get("original_query", "")):
        from src.gateway.llm_gateway import llm_gateway as gateway
        messages = build_code_messages(query, state.get("true_data_chunks", []), catalog,
                                       state.get("conversation_history", []))
        try:
            model = gateway.code_model_string()
            response = gateway.complete(messages, temperature=0.2, max_tokens=2000, model=model)
        except Exception as exc:
            logger.warning("code_model_unavailable_using_default", error=str(exc)[:200])
            model = gateway._build_model_string()
            response = gateway.complete(messages, temperature=0.2, max_tokens=2000) + CODE_MODEL_HINT
        steps.append("generation")
        return {"llm_response": response, "model_used": model, "pipeline_steps": steps}

    if not state.get("true_data_chunks"):
        # Nothing relevant was retrieved: a small model would answer from general
        # knowledge and invent citations, so answer honestly without calling it.
        steps.append("generation")
        pending = unindexed_sources(state.get("source_filter"))
        no_history = missing_commit_history(state.get("original_query", ""), state.get("source_filter"))
        return {"llm_response": no_context_answer(catalog, state.get("original_query", ""), pending, no_history),
                "model_used": "none (no matching sources)",
                "pipeline_steps": steps}

    direct = commit_list_answer(state.get("original_query", ""), state.get("true_data_chunks", []))
    if direct:
        steps.append("generation")
        return {"llm_response": direct, "model_used": "none (commit history)", "pipeline_steps": steps}

    messages = build_messages(query, context, catalog, state.get("conversation_history", []),
                              state.get("original_query", ""))

    try:
        response = llm_gateway.complete(messages, temperature=0.1, max_tokens=1500, stop=llm_gateway.ANSWER_STOP)
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
