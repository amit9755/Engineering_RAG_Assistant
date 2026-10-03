"""Create Jira tickets from a chat: draft from the conversation, review in a form, then create.

Nothing is created automatically: the chat produces a draft (local model), the
user reviews / edits it in a form that shows the project's real issue types and
required fields (read from Jira), and only "Create in Jira" calls the API.
Tickets are created with the saved Jira token, so its owner is the reporter.
"""

import json
import re
from typing import Dict, List, Optional

from src.sources.jobs import IndexingError
from src.observability.logger import get_logger

logger = get_logger(__name__)

CREATE_INTENT = re.compile(
    r"\b(?:create|raise|open|file|log|make|add|submit)\s+(?:a\s+|an\s+|new\s+|the\s+)*"
    r"(?:jira\s*)?(?:ticket|issue|bug|story|task|jira)\b", re.I)

PREFERRED_TYPES = ["Bug", "Task", "Story"]
# Fields the form always shows; other required fields are rendered from Jira's metadata.
STANDARD_FIELDS = {"project", "issuetype", "summary", "description", "reporter", "priority", "labels",
                   "attachment", "issuelinks"}

DRAFT_PROMPT = """You write Jira tickets for an engineering team. From the conversation below, draft ONE
ticket for what the user asked to create. Reply with JSON only, no other text:
{{"issue_type": "Bug" | "Task" | "Story",
  "summary": "<one line, max 120 characters, specific>",
  "description": "<plain text: what is wrong or needed, context, steps to reproduce for bugs, expected
                   vs actual, acceptance criteria for stories/tasks, suggested fix if discussed>",
  "priority": "Highest" | "High" | "Medium" | "Low" | "Lowest",
  "labels": ["<short-label>", ...]}}
Use Bug for errors and failures, Story for new user-facing features, Task otherwise.
Only include facts from the conversation; do not invent ticket numbers, people or dates.

Conversation:
{conversation}

Request: {question}"""


_ABOUT_CODE = re.compile(r"\b(?:write|generate|suggest|implement|show)\b.{0,40}\b(?:code|script|function)\b"
                         r"|\b(?:code|script|function) (?:to|for|that)\b", re.I)


def is_create_request(question: str) -> bool:
    """"create a jira ticket for this" - but not "write code to create a jira ticket"."""
    return bool(CREATE_INTENT.search(question or "")) and not _ABOUT_CODE.search(question or "")


def _parse_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except ValueError:
        return {}


def draft_ticket(question: str, history: List[Dict], sources: List[str] = None) -> dict:
    """A draft {issue_type, summary, description, priority, labels} written by the local model."""
    from src.gateway.llm_gateway import llm_gateway
    prior = history[:-1] if history and history[-1].get("content") == question else history
    conversation = "\n\n".join(f"{m['role'].upper()}: {m['content'][:2500]}" for m in prior[-6:]) or "(none)"
    draft = {}
    try:
        draft = _parse_json(llm_gateway.complete(
            [{"role": "user", "content": DRAFT_PROMPT.format(conversation=conversation, question=question)}],
            temperature=0.1, max_tokens=700))
    except Exception as exc:
        logger.warning("jira_draft_failed", error=str(exc)[:200])
    last_answer = next((m["content"] for m in reversed(prior) if m.get("role") == "assistant"), "")
    issue_type = str(draft.get("issue_type") or "")
    if issue_type not in PREFERRED_TYPES:
        issue_type = "Bug" if re.search(r"\b(bug|error|fail|broken|crash|exception)\w*", question + last_answer, re.I) \
            else "Task"
    summary = " ".join(str(draft.get("summary") or question).split())[:200]
    description = str(draft.get("description") or last_answer or question).strip()
    if sources:
        description += "\n\nRelated (from the Engineering RAG Assistant chat):\n" + "\n".join(
            f"- {s}" for s in sources[:10])
    labels = [re.sub(r"\s+", "-", str(l).strip())[:40] for l in (draft.get("labels") or []) if str(l).strip()][:5]
    return {"issue_type": issue_type, "summary": summary, "description": description,
            "priority": str(draft.get("priority") or "Medium"), "labels": labels}


def _client(source_id: str):
    from src.sources.credentials import credential_store
    from src.sources.jira_indexer import JiraClient
    from src.sources.registry import source_registry
    cfg = source_registry.get_jira_config(source_id)
    if not cfg:
        raise IndexingError("Jira project not found. Add it under Knowledge Sources > Jira.")
    email, token = credential_store.retrieve(cfg.credential_id).split(":", 1)
    return JiraClient(cfg.base_url, email, token), cfg


def _field_spec(field: dict) -> dict:
    schema = field.get("schema") or {}
    kind, items = schema.get("type"), schema.get("items")
    if kind in ("option", "priority", "version", "component", "resolution") or (
            kind == "array" and items in ("option", "version", "component")):
        ftype = "multi-option" if kind == "array" else "option"
    elif kind in ("string", "number", "date", "datetime"):
        ftype = kind
    elif kind == "user":
        ftype = "user"
    elif kind == "array" and items == "string":
        ftype = "labels"
    else:
        ftype = "unsupported"
    options = [{"id": str(v.get("id") or v.get("value") or v.get("name")),
                "name": v.get("name") or v.get("value") or str(v.get("id"))}
               for v in field.get("allowedValues") or []]
    return {"id": field.get("fieldId") or field.get("key") or field.get("id"), "name": field.get("name", ""),
            "required": bool(field.get("required")) and not field.get("hasDefaultValue"),
            "type": ftype, "options": options}


def issue_types(source_id: str) -> List[dict]:
    """Non-subtask issue types of the project, Bug / Task / Story first."""
    client, cfg = _client(source_id)
    response = client.api.get(f"/rest/api/2/issue/createmeta/{cfg.project_key}/issuetypes", {"maxResults": 100})
    if response.status_code == 200:
        types = client.api.json(response, "issue types").get("values", [])
    else:   # older Jira Server: the combined createmeta endpoint
        response = client.api.get("/rest/api/2/issue/createmeta", {"projectKeys": cfg.project_key})
        projects = client.api.json(response, "issue types").get("projects", []) if response.status_code == 200 else []
        types = projects[0].get("issuetypes", []) if projects else []
    types = [{"id": str(t["id"]), "name": t["name"]} for t in types if not t.get("subtask")]
    if not types:
        raise IndexingError(f"No issue types are available to create in {cfg.project_key}; check that your token "
                            "has the Create Issues permission.")
    order = {name: i for i, name in enumerate(PREFERRED_TYPES)}
    return sorted(types, key=lambda t: order.get(t["name"], len(order)))


def create_fields(source_id: str, issue_type_id: str) -> List[dict]:
    """Field specs for an issue type: priority and every other required field the form must ask for."""
    client, cfg = _client(source_id)
    response = client.api.get(f"/rest/api/2/issue/createmeta/{cfg.project_key}/issuetypes/{issue_type_id}",
                              {"maxResults": 200})
    if response.status_code == 200:
        fields = client.api.json(response, "fields").get("values", [])
    else:
        response = client.api.get("/rest/api/2/issue/createmeta", {
            "projectKeys": cfg.project_key, "issuetypeIds": issue_type_id, "expand": "projects.issuetypes.fields"})
        data = client.api.json(response, "fields") if response.status_code == 200 else {}
        types = (data.get("projects") or [{}])[0].get("issuetypes") or [{}]
        fields = [{**v, "fieldId": k} for k, v in (types[0].get("fields") or {}).items()]
    specs = [_field_spec(f) for f in fields]
    return [s for s in specs if s["id"] == "priority" or (s["required"] and s["id"] not in STANDARD_FIELDS)]


def _field_value(spec_type: str, value, cloud: bool):
    if spec_type == "option":
        return {"id": str(value)}
    if spec_type == "multi-option":
        return [{"id": str(v)} for v in (value if isinstance(value, list) else [value])]
    if spec_type == "number":
        return float(value)
    if spec_type == "user":
        return {"accountId": value} if cloud else {"name": value}
    if spec_type == "labels":
        return [v.strip() for v in str(value).split(",") if v.strip()]
    return value


def create_issue(source_id: str, issue_type_id: str, summary: str, description: str,
                 labels: List[str] = None, fields: Optional[Dict] = None, created_by: str = "") -> dict:
    """Create the ticket; returns {key, url}. Jira's own validation messages are passed on."""
    client, cfg = _client(source_id)
    summary = " ".join((summary or "").split())
    if not summary:
        raise IndexingError("A summary is required.")
    specs = {s["id"]: s for s in create_fields(source_id, issue_type_id)}
    body = {"project": {"key": cfg.project_key}, "issuetype": {"id": str(issue_type_id)},
            "summary": summary[:250], "description": description or ""}
    if labels:
        body["labels"] = [re.sub(r"\s+", "-", l.strip()) for l in labels if l.strip()]
    for field_id, value in (fields or {}).items():
        if value in (None, "", []):
            continue
        spec = specs.get(field_id, {"type": "string"})
        if spec["type"] == "unsupported":
            continue
        body[field_id] = _field_value(spec["type"], value, client.api.cloud)
    response = client.api.post("/rest/api/2/issue", {"fields": body})
    if response.status_code not in (200, 201):
        try:
            data = client.api.json(response, "create issue")
        except IndexingError:
            data = {}
        problems = list(data.get("errorMessages") or []) + [
            f"{specs.get(k, {}).get('name') or k}: {v}" for k, v in (data.get("errors") or {}).items()]
        raise IndexingError("Jira did not create the ticket: " + ("; ".join(problems) or f"HTTP {response.status_code}"))
    key = client.api.json(response, "create issue")["key"]
    logger.info("jira_issue_created", key=key, project=cfg.project_key, created_by=created_by)
    return {"key": key, "url": f"{cfg.base_url}/browse/{key}"}
