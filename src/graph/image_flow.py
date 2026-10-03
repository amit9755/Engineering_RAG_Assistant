"""Questions with images: read the image once, find where its text appears in the code, answer.

The vision model only transcribes and briefly describes the image (short output,
one pass), because on a CPU-only PC long vision answers take minutes. Distinctive
text from the image - error messages, button labels, titles, file paths, names -
usually appears literally in the code, so it is searched like "find in files".
"Where is the code for this screenshot?" is answered straight from those matches;
other questions are answered by the faster text model using what the image shows.
"""

import re
from typing import Dict, List, Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)

READ_PROMPT = (
    "You are helping a developer. Copy every piece of visible text in this image exactly as written, "
    "one item per line: titles, headings, labels, buttons, menu items, column headers, messages, errors, "
    "codes, file paths, URLs, names and IDs. Then add one final line starting with 'Image shows:' that says "
    "briefly what the image is (for example a web page, dialog, terminal output, log or diagram).")

LOCATE_INTENT = re.compile(
    r"\b(where|which (file|files|component|components|module|function|class|page)|find|locate|search|look for|"
    r"related code|source code|code (for|behind|of|related)|implemented|implementation|defined|handled)\b", re.I)

_STOP = {"the", "and", "for", "with", "this", "that", "from", "image", "shows", "page", "button", "menu",
         "error", "ok", "cancel", "yes", "no", "close", "save", "submit", "search", "home", "next", "back"}
_IDENTIFIER = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
_PATH = re.compile(r"[\w./-]+\.(?:py|js|jsx|ts|tsx|java|go|rb|php|cs|cpp|c|h|html|css|scss|sql|json|ya?ml|md)\b")
_QUOTED = re.compile(r"[\"'“‘`]([^\"'”’`]{4,80})[\"'”’`]")


class ImageModelError(Exception):
    """Ollama refused the image request; the message says what to do."""


def _ollama_base() -> str:
    import os
    from dotenv import dotenv_values
    return (os.environ.get("OLLAMA_API_BASE") or dotenv_values(".env").get("OLLAMA_API_BASE")
            or "http://localhost:11434").rstrip("/")


def explain_ollama_error(model: str, status: int, message: str) -> str:
    """Ollama's own error, turned into an instruction."""
    text = (message or "").lower()
    if status == 404 or "not found" in text or "pull" in text:
        return f"The image model **{model}** is not installed in Ollama. Run `ollama pull {model}`, then ask again."
    if "unknown model architecture" in text or "unsupported" in text or "does not support" in text:
        return (f"This Ollama version cannot run **{model}**. Update Ollama (download the latest from "
                "https://ollama.com/download, or `winget upgrade Ollama.Ollama`), restart it, then ask again.")
    if "memory" in text or "out of memory" in text or "oom" in text:
        return (f"Not enough memory to load **{model}** next to the other models. Close other programs or "
                "restart Ollama, then ask again.")
    return f"Ollama could not read the image (HTTP {status}): {message[:300]}"


def read_image(images_b64: List[str], model: str) -> str:
    """
    Transcription + one-line description of the images (one short vision pass).
    Calls Ollama's chat API directly: LiteLLM hides Ollama's error text when a
    streamed request fails ("<generator object Response.iter_lines ...>").
    """
    import os
    import requests
    from src import network_policy
    name = model.split("/", 1)[-1]
    url = f"{_ollama_base()}/api/chat"
    network_policy.check_url(url, "the Ollama model server")
    body = {"model": name, "stream": False, "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": 400, "num_ctx": 8192},
            "messages": [{"role": "user", "content": READ_PROMPT, "images": images_b64}]}
    try:
        response = requests.post(url, json=body, timeout=int(os.environ.get("LLM_TIMEOUT", "600")))
    except requests.exceptions.Timeout as exc:
        raise ImageModelError("Reading the image took too long. Try a smaller screenshot, or raise LLM_TIMEOUT "
                              "in .env.") from exc
    except requests.RequestException as exc:
        raise ImageModelError("Cannot reach Ollama. Make sure Ollama is running (open the Ollama app or run "
                              "`ollama serve`).") from exc
    if response.status_code != 200:
        try:
            message = response.json().get("error", response.text)
        except ValueError:
            message = response.text
        logger.warning("ollama_image_error", status=response.status_code, error=message[:300])
        raise ImageModelError(explain_ollama_error(name, response.status_code, message))
    return ((response.json().get("message") or {}).get("content") or "").strip()


def extract_search_terms(text: str, limit: int = 14) -> List[str]:
    """Distinctive strings from the transcription, most specific first."""
    terms = []

    def add(term):
        term = term.strip(" \t-*•:|,;.")
        if 4 <= len(term) <= 80 and term.lower() not in _STOP and term not in terms:
            terms.append(term)

    for match in _PATH.finditer(text):
        add(match.group(0).split("/")[-1])          # file name, e.g. publish.ts
    for match in _IDENTIFIER.finditer(text):
        add(match.group(0))                          # camelCase, snake_case, CONSTANT_NAME
    for match in _QUOTED.finditer(text):
        add(match.group(1))
    for line in text.splitlines():
        line = line.strip(" \t-*•")
        if not line or line.lower().startswith("image shows"):
            continue
        words = line.split()
        if 2 <= len(words) <= 12 and re.search(r"[A-Za-z]{3}", line):
            if len(words) <= 8:
                add(line)                            # labels, titles, short messages
            if ":" in line:
                add(line.split(":", 1)[0])           # "Last checked: Never" -> "Last checked"
                add(line.split(":", 1)[1])           # "Error 190: Invalid OAuth token" -> message part
            if len(words) >= 5:
                # One misread word (OCR) must not sink the whole line: also try its start and end.
                add(" ".join(words[:3]))
                add(" ".join(words[-3:]))
    return terms[:limit]


def is_weak_term(term: str) -> bool:
    """A single ordinary word ("Never", "Status"): matches everywhere, so it cannot locate code alone."""
    return " " not in term and not (_IDENTIFIER.fullmatch(term) or _PATH.fullmatch(term)
                                     or re.search(r"\d|[._/-]", term))


def _variants(term: str) -> List[str]:
    """UI text is often styled (CSS uppercase / capitalize): try the casings code usually uses."""
    out = []
    for v in (term, term.lower(), term.title(), term.capitalize()):
        if v not in out:
            out.append(v)
    return out


_HEADER = re.compile(r"^(Repository|File|Confluence page|Path|Last updated|Jira issue):", re.I)


def _best_line(texts: List[str], terms: List[str]) -> str:
    """The content line containing the most matched terms (not a chunk's Repository / File header)."""
    best, best_count = "", 0
    for text in texts:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or _HEADER.match(stripped):
                continue
            count = sum(1 for t in terms if t.lower() in stripped.lower())
            if count > best_count:
                best, best_count = stripped[:140], count
    return best


def find_code_matches(terms: List[str], source_filter=None, limit: int = 10) -> List[Dict]:
    """Files whose indexed text contains the terms; files matching more terms rank first."""
    from src.retrieval.vector_store import vector_store
    files: Dict[str, Dict] = {}
    for term in terms:
        hits = []
        for variant in _variants(term):
            hits = vector_store.find_text(variant, source_filter)
            if hits:
                break
        for doc in hits:
            meta = doc.metadata
            key = meta.get("source_file") or meta.get("file_path") or "unknown"
            if key.endswith("(commit history)") or key.endswith("(file tree)"):
                continue
            entry = files.setdefault(key, {"file": key, "url": meta.get("url", ""), "terms": [], "line": "",
                                           "source_type": meta.get("source_type", "document"), "texts": []})
            if term not in entry["terms"]:
                entry["terms"].append(term)
            if doc.page_content not in entry["texts"] and len(entry["texts"]) < 4:
                entry["texts"].append(doc.page_content)
    for entry in files.values():
        entry["line"] = _best_line(entry.pop("texts"), entry["terms"])
    # Files that matched only single ordinary words ("Never") are noise once anything
    # matched a distinctive term (a label, message or identifier).
    if any(not is_weak_term(t) for e in files.values() for t in e["terms"]):
        files = {k: e for k, e in files.items() if any(not is_weak_term(t) for t in e["terms"])}
    def rank(entry):
        path = entry["file"].lower()
        is_test = any(t in path for t in ("/test", "test/", ".test.", ".spec.", "__tests__"))
        is_docs = entry["source_type"] != "bitbucket" or path.endswith((".md", ".rst", ".txt", ".adoc"))
        # Code first (docs mention the same words), then most matched terms, then non-test files.
        return (is_docs, -len(entry["terms"]), is_test, len(path))
    return sorted(files.values(), key=rank)[:limit]


def _short(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[2:]) if len(parts) >= 3 else path   # drop "workspace/repo/" prefix


def format_locate_answer(matches: List[Dict], retrieved: List[str], transcription: str) -> str:
    seen = "\n".join(f"- {l.strip(' -*•')}" for l in transcription.splitlines() if l.strip())[:1500]
    matched = {m["file"] for m in matches}
    related = [f for f in retrieved if f not in matched and not f.endswith(("(commit history)", "(file tree)"))][:6]
    if matches:
        rows = "\n".join(
            f"| {i} | {'[' + _short(m['file']) + '](' + m['url'] + ')' if m['url'].startswith('http') else '`' + _short(m['file']) + '`'} "
            f"| {', '.join(m['terms'][:4]).replace('|', '/')} | `{m['line'].replace('|', '/').replace('`', '')}` |"
            for i, m in enumerate(matches, 1))
        head = (f"### Where this appears in the code\n\nFound text from the image in **{len(matches)} file"
                f"{'s' if len(matches) != 1 else ''}** (most matches first):\n\n"
                "| # | File | Matched text | Line |\n|---|---|---|---|\n" + rows)
        if related:
            head += "\n\n**Related by meaning** (no exact text match):\n" + "\n".join(f"- `{_short(f)}`" for f in related)
    else:
        head = ("### Where this appears in the code\n\nNone of the text in the image appears word-for-word in "
                "the selected sources.")
        if related:
            head += " The closest related files by meaning are:\n\n" + "\n".join(
                f"- `{_short(f)}`" for f in related)
    return head + f"\n\n**Text read from the image:**\n{seen}"


IMAGE_ANSWER_PROMPT = """You are a precise engineering assistant. The user attached an image. A vision
model read it; its transcription is below. Answer the user's question using the transcription, the exact
code matches, and the numbered sources.

How to answer:
- Quote error text, codes and names exactly as transcribed.
- When the matches or sources show where something is implemented, name the file and function.
- Cite sources by file path or Jira key. If the sources are unrelated, answer from the image and say so.
- If something is unclear in the transcription, say so instead of guessing.

What the image contains (read by the vision model):
{transcription}

Files where text from the image appears exactly:
{matches}

Knowledge sources the user selected:
{catalog}

Sources:
{context}"""


def build_answer_messages(question: str, transcription: str, matches: List[Dict], context: str,
                          catalog: List[str], history: List[Dict]) -> List[Dict]:
    match_text = "\n".join(f"- {m['file']}: {', '.join(m['terms'][:4])} -> {m['line']}" for m in matches) or "(none)"
    messages = [{"role": "system", "content": IMAGE_ANSWER_PROMPT.format(
        transcription=transcription or "(nothing readable)", matches=match_text,
        catalog="\n".join(catalog) or "(none)", context=context or "(no matching sources)")}]
    if history and history[-1].get("content") == question:
        history = history[:-1]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-4:]]
    messages.append({"role": "user", "content": question})
    return messages


def is_locate_question(question: str) -> bool:
    return bool(LOCATE_INTENT.search(question or ""))
