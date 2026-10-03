"""Answer Jira filter questions with a live JQL search instead of text search.

"all jira created by me last month", "issues resolved by me in the last 6
months", "how many open bugs assigned to Naresh" are filters on reporter /
assignee / resolver, dates, status and type. Similarity search cannot answer
them (it finds similar text, not "reporter = me"), so they are translated into
JQL and run against Jira with the saved token. "me" is the token's owner
(currentUser()). The answer is an exact table plus the JQL and a Jira link.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import quote

from src.observability.logger import get_logger

logger = get_logger(__name__)

MAX_LISTED = 50
MAX_COUNTED = 1000
FIELDS = "summary,status,issuetype,priority,reporter,assignee,created,resolutiondate,resolution"

_JIRA_WORDS = re.compile(r"\b(jiras?|issues?|tickets?|bugs?|defects?|stor(y|ies)|tasks?|sub-?tasks?|epics?)\b", re.I)
_FILTER_CUES = re.compile(
    r"\b(my|mine|me|created|create|creat\w*|reported|raised|opened|filed|logged|assigned|resolved|resolve|closed|"
    r"fixed|completed|done|open|unresolved|pending|last|past|previous|this|today|yesterday|recent|latest|"
    r"how many|count|number of|list|all|show)\b", re.I)
_ISSUE_KEY = re.compile(r"\b[A-Z][A-Z0-9_]+-\d+\b")
# "my", "by me", "assigned to me", "I created" - but not "tell me" / "give me" / "show me".
_ME = re.compile(r"\b(by me|my|mine|to me|for me|myself|i (?:have |had )?(?:created|reported|raised|opened|"
                 r"resolved|closed|fixed|worked|filed|logged))\b", re.I)
# A name right after the issue word ("jira Deversh jani", "tickets of Naresh") or possessive ("Naresh's bugs").
_NAMED = re.compile(r"\b(?:jiras?|issues?|tickets?|bugs?|tasks?|stor(?:y|ies)|defects?)\s+(?:of|for|from|by|"
                    r"assigned to|created by|reported by|raised by)?\s*([A-Za-z][a-z]+(?:\s+[A-Za-z][a-z]+){0,2})", re.I)
_POSSESSIVE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'s\s+(?:jiras?|issues?|tickets?|bugs?|tasks?)", re.I)
_COMMON = {"open", "opened", "closed", "created", "create", "resolved", "fixed", "done", "pending", "last", "past",
           "this", "that", "these", "all", "any", "in", "on", "the", "from", "which", "with", "about", "and", "or",
           "assigned", "reported", "raised", "status", "list", "show", "give", "tell", "me", "my", "is", "are",
           "were", "was", "today", "yesterday", "week", "month", "year", "recent", "latest", "count", "how", "many",
           "high", "low", "priority", "unresolved", "still", "not", "to", "of", "for", "by", "jira", "issues"}
_PERSON = re.compile(r"\b(?:by|to|for)\s+([A-Za-z][\w.'-]*(?:\s+[A-Za-z][\w.'-]*){0,2}?)"
                     r"(?=\s+(?:in|on|during|from|since|last|past|this|which|that|and|with|about|for)\b|[?.!,]|$)", re.I)
_NOT_A_NAME = {"me", "my", "mine", "myself", "status", "date", "priority", "type", "project", "jira", "issue",
               "issues", "month", "week", "year", "day", "days", "months", "weeks", "years", "the", "a", "all"}
_UNITS = {"day": 1, "week": 7, "month": 30, "year": 365}
_TYPES = [("bug", "Bug"), ("defect", "Bug"), ("stor", "Story"), ("sub-task", "Sub-task"), ("subtask", "Sub-task"),
          ("task", "Task"), ("epic", "Epic")]


@dataclass
class JiraQuery:
    person: Optional[str] = None      # "me", a name, or None
    role: str = "any"                 # reporter | assignee | resolver | any
    status: Optional[str] = None      # open | resolved
    since: Optional[str] = None       # JQL date expression, e.g. "-30d" or "startOfMonth()"
    period_label: str = ""
    date_field: str = "created"
    issue_types: List[str] = field(default_factory=list)
    text: Optional[str] = None
    limit: int = MAX_LISTED
    count_only: bool = False


def parse_jira_question(question: str) -> Optional[JiraQuery]:
    """A JiraQuery for filter-style Jira questions, else None (left to normal search)."""
    q = " ".join((question or "").split())
    if not _JIRA_WORDS.search(q) or _ISSUE_KEY.search(q):
        return None
    if not (_FILTER_CUES.search(q) or _POSSESSIVE.search(q) or _NAMED.search(q)):
        return None
    lower = q.lower()
    query = JiraQuery()

    if _ME.search(q):
        query.person = "me"
    else:
        candidates = [m.group(1) for m in _PERSON.finditer(q)] + [m.group(1) for m in _POSSESSIVE.finditer(q)] + \
                     [m.group(1) for m in _NAMED.finditer(q)]
        for raw in candidates:
            words = []
            for word in raw.split():
                if word.lower() in _COMMON or word.lower() in _NOT_A_NAME:
                    break          # a name ends where ordinary words start ("Naresh in last month")
                words.append(word)
            if words and len(" ".join(words)) >= 3:
                query.person = " ".join(words)
                break

    if re.search(r"\b(resolv\w*|closed|close|fixed|fix|completed|done)\b", lower) and query.person:
        query.role = "resolver"
    elif re.search(r"\b(creat\w*|reported|report|raised|raise|opened|filed|logged)\b", lower):
        query.role = "reporter"
    elif re.search(r"\b(assigned|assignee|working on|work on)\b", lower):
        query.role = "assignee"

    if query.role != "resolver":
        if re.search(r"\b(open|unresolved|pending|not resolved|in progress|still)\b", lower):
            query.status = "open"
        elif re.search(r"\b(resolv\w*|closed|fixed|completed|done)\b", lower):
            query.status = "resolved"

    match = re.search(r"\b(?:last|past|previous)\s+(\d+)?\s*(day|week|month|year)s?\b", lower)
    if match:
        n = int(match.group(1) or 1)
        query.since = f"-{n * _UNITS[match.group(2)]}d"
        query.period_label = f"in the last {n} {match.group(2)}s" if n > 1 else f"in the last {match.group(2)}"
    elif re.search(r"\bthis (week|month|year)\b", lower):
        unit = re.search(r"\bthis (week|month|year)\b", lower).group(1)
        query.since, query.period_label = f"startOf{unit.title()}()", f"this {unit}"
    elif "today" in lower:
        query.since, query.period_label = "startOfDay()", "today"
    elif "yesterday" in lower:
        query.since, query.period_label = "startOfDay(-1)", "since yesterday"

    if query.role == "resolver" or (query.status == "resolved" and query.role == "any"):
        query.date_field = "resolved"
    elif "updated" in lower:
        query.date_field = "updated"

    for word, issue_type in _TYPES:
        if re.search(rf"\b{word}", lower) and issue_type not in query.issue_types:
            query.issue_types.append(issue_type)

    about = re.search(r"\b(?:about|regarding|related to|mentioning|containing)\s+(.+?)[?.!]*$", q, re.I)
    if about:
        query.text = about.group(1).strip().strip('"')

    count = re.search(r"\b(?:last|latest|recent|top|first)\s+(\d{1,3})\b(?!\s*(?:day|week|month|year))", lower)
    if count:
        query.limit = min(int(count.group(1)), MAX_LISTED)
    query.count_only = bool(re.search(r"\b(how many|count|number of)\b", lower))

    # A bare "jira issues" mention with no person, status, period or type is not a filter question.
    if not (query.person or query.status or query.since or query.issue_types or query.role != "any"):
        return None
    return query


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_jql(project_key: str, query: JiraQuery, user: str) -> str:
    """user: 'currentUser()' or a quoted user name / account id."""
    clauses = [f"project = {_quote(project_key)}"]
    if user:
        if query.role == "reporter":
            clauses.append(f"reporter = {user}")
        elif query.role == "assignee":
            clauses.append(f"assignee = {user}")
        elif query.role == "resolver":
            # Jira has no "resolved by" field; the change history of the resolution records who resolved it.
            during = f" DURING ({query.since}, now())" if query.since else ""
            clauses.append(f"resolution CHANGED BY {user}{during}")
            clauses.append("resolution != Unresolved")
        else:
            clauses.append(f"(reporter = {user} OR assignee = {user})")
    if query.status == "open":
        clauses.append("resolution = Unresolved")
    elif query.status == "resolved":
        clauses.append("resolution != Unresolved")
    if query.since and query.role != "resolver":
        clauses.append(f"{query.date_field} >= {query.since}")
    if query.issue_types:
        clauses.append("issuetype in (" + ", ".join(_quote(t) for t in query.issue_types) + ")")
    if query.text:
        clauses.append(f"text ~ {_quote(query.text)}")
    order = {"resolved": "resolved", "updated": "updated"}.get(query.date_field, "created")
    return " AND ".join(clauses) + f" ORDER BY {order} DESC"


def _who(query: JiraQuery) -> str:
    person = "you" if query.person == "me" else query.person
    if not person:
        return ""
    return {"reporter": f" reported by {person}", "assignee": f" assigned to {person}",
            "resolver": f" resolved by {person}"}.get(query.role, f" reported by or assigned to {person}")


def format_answer(project_key: str, query: JiraQuery, jql: str, issues: list, total: int, base_url: str) -> str:
    kinds = "/".join(query.issue_types).lower() + "s" if query.issue_types else "issues"
    status = {"open": "open ", "resolved": "resolved "}.get(query.status or "", "")
    period = f" ({'resolved' if query.role == 'resolver' else query.date_field} {query.period_label})" \
        if query.period_label else ""
    title = f"**{total} {status}{kinds}** in {project_key}{_who(query)}{period}"
    link = f"{base_url}/issues/?jql={quote(jql)}"
    if not total:
        return f"{title}.\n\nJQL used: `{jql}`\n\n[Open this search in Jira]({link})"
    if query.count_only:
        return f"{title}.\n\nJQL used: `{jql}`\n\n[Open this search in Jira]({link})"
    rows = []
    for n, issue in enumerate(issues[:query.limit], 1):
        f = issue.get("fields") or {}
        rows.append("| {n} | [{key}]({url}) | {summary} | {status} | {reporter} | {assignee} | {created} | {resolved} |"
                    .format(n=n, key=issue["key"], url=f"{base_url}/browse/{issue['key']}",
                            summary=(f.get("summary") or "").replace("|", "/")[:120],
                            status=(f.get("status") or {}).get("name", ""),
                            reporter=(f.get("reporter") or {}).get("displayName", ""),
                            assignee=(f.get("assignee") or {}).get("displayName", "Unassigned"),
                            created=(f.get("created") or "")[:10], resolved=(f.get("resolutiondate") or "")[:10]))
    shown = f" (showing the newest {len(rows)})" if total > len(rows) else ""
    return (f"{title}{shown}:\n\n| # | Issue | Summary | Status | Reporter | Assignee | Created | Resolved |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) +
            f"\n\nJQL used: `{jql}`\n\n[Open this search in Jira]({link})  (live from Jira, not the index)")


def _resolve_user(client, name: str) -> Optional[str]:
    """JQL user reference for a display name, via Jira's user search."""
    cloud = client.api.cloud
    path, params = ("/rest/api/3/user/search", {"query": name}) if cloud else \
        ("/rest/api/2/user/search", {"username": name, "maxResults": 5})
    tokens = [t.lower() for t in name.split()]
    users = []
    # Full name first; many servers only prefix-match one word, so fall back to the first name.
    for term in dict.fromkeys([name, name.split()[0]]):
        params = {"query": term} if cloud else {"username": term, "maxResults": 20}
        response = client._get(path, params)
        if response.status_code == 200:
            users = client.api.json(response, "user search") or []
        if users:
            break
    def label(u):
        return f"{u.get('displayName') or ''} {u.get('name') or ''} {u.get('emailAddress') or ''}".lower()
    matching = [u for u in users if all(t in label(u) for t in tokens)]
    if not matching:
        return None
    exact = [u for u in matching if (u.get("displayName") or "").lower() == name.lower()]
    user = (exact or matching)[0]
    return _quote(user.get("accountId") or user.get("name") or user.get("key"))


def answer_jira_question(question: str, source_filter=None) -> Optional[Tuple[str, List[str]]]:
    """(answer, source labels) for a Jira filter question on the selected Jira projects; None otherwise."""
    query = parse_jira_question(question)
    if not query:
        return None
    from src.sources.registry import source_registry
    from src.sources.credentials import credential_store
    from src.sources.jira_indexer import JiraClient
    from src.sources.jobs import IndexingError

    projects = []
    for src in source_registry.list_sources():
        if src.type.value != "jira" or (source_filter is not None and src.id not in source_filter.source_ids):
            continue
        cfg = source_registry.get_jira_config(src.id)
        if cfg:
            projects.append(cfg)
    if not projects:
        return None

    parts, labels = [], []
    for cfg in projects:
        try:
            email, token = credential_store.retrieve(cfg.credential_id).split(":", 1)
            client = JiraClient(cfg.base_url, email, token)
            user = None
            if query.person == "me":
                user = "currentUser()"
            elif query.person:
                user = _resolve_user(client, query.person)
                if not user:
                    parts.append(f"No Jira user matching **{query.person}** was found on {cfg.base_url}.")
                    continue
            jql = build_jql(cfg.project_key, query, user)
            limit = MAX_COUNTED if query.count_only else query.limit
            issues, total = client.search(jql, limit)
            logger.info("jira_live_query", project=cfg.project_key, jql=jql, total=total)
            parts.append(format_answer(cfg.project_key, query, jql, issues, total, cfg.base_url))
            labels += [i["key"] for i in issues[:query.limit]]
        except (IndexingError, KeyError, ValueError) as exc:
            parts.append(f"Couldn't run the live Jira search on {cfg.project_key}: {exc}")
    return "\n\n".join(parts), labels
