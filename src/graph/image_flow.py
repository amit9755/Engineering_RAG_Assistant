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


def read_image(images_b64: List[str], model: str) -> str:
    """Transcription + one-line description of the images (one short vision pass)."""
    from src.gateway.llm_gateway import llm_gateway
    return llm_gateway.complete_via_stream(
        [{"role": "user", "content": READ_PROMPT, "images": images_b64}],
        temperature=0.0, max_tokens=400, model=model).strip()


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
        if 2 <= len(words) <= 8 and re.search(r"[A-Za-z]{3}", line):
            add(line)                                # labels, titles, short messages
            if ":" in line:
                add(line.split(":", 1)[1])           # "Error 190: Invalid OAuth token" -> message part
    return terms[:limit]


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
        hits = vector_store.find_text(term, source_filter)
        if not hits and term != term.lower():
            hits = vector_store.find_text(term.lower(), source_filter)
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
    if matches:
        rows = "\n".join(
            f"| {i} | {'[' + _short(m['file']) + '](' + m['url'] + ')' if m['url'].startswith('http') else '`' + _short(m['file']) + '`'} "
            f"| {', '.join(m['terms'][:4]).replace('|', '/')} | `{m['line'].replace('|', '/').replace('`', '')}` |"
            for i, m in enumerate(matches, 1))
        head = (f"### Where this appears in the code\n\nFound text from the image in **{len(matches)} file"
                f"{'s' if len(matches) != 1 else ''}** (most matches first):\n\n"
                "| # | File | Matched text | Line |\n|---|---|---|---|\n" + rows)
    else:
        head = ("### Where this appears in the code\n\nNone of the text in the image appears word-for-word in "
                "the selected sources.")
        if retrieved:
            head += " The closest related files by meaning are:\n\n" + "\n".join(
                f"- `{_short(f)}`" for f in retrieved[:6])
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
