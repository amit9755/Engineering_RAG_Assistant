"""Live JQL answers for Jira filter questions (offline: Jira is mocked)."""

from unittest.mock import Mock

import pytest

from src.retrieval.source_filter import SourceFilter
from src.sources.jira_query import answer_jira_question, build_jql, parse_jira_question
from src.sources.models import JiraSourceConfig, SourceStatus
from tests.test_connectors import FakeCredentials, registry, response  # noqa: F401


@pytest.mark.parametrize("question,jql", [
    ("give me last month all jira which is create by me mean Amit kushwah",
     'project = "P" AND reporter = currentUser() AND created >= -30d ORDER BY created DESC'),
    ("last 6 month all my jira which is resolved my me",
     'project = "P" AND resolution CHANGED BY currentUser() DURING (-180d, now()) AND resolution != Unresolved '
     'ORDER BY resolved DESC'),
    ("show my open jira",
     'project = "P" AND (reporter = currentUser() OR assignee = currentUser()) AND resolution = Unresolved '
     'ORDER BY created DESC'),
    ("list jira issues created this week", 'project = "P" AND created >= startOfWeek() ORDER BY created DESC'),
    ("list all bugs about reservation timeout",
     'project = "P" AND issuetype in ("Bug") AND text ~ "reservation timeout" ORDER BY created DESC'),
])
def test_questions_become_jql(question, jql):
    query = parse_jira_question(question)
    assert build_jql("P", query, "currentUser()" if query.person == "me" else None) == jql


@pytest.mark.parametrize("question", [
    "what is WSQAAUTO-21777 about?", "which jira issues talk about login timeout", "explain this project",
    "tell me last 5 commits", "how does the reservation flow work?",
])
def test_non_filter_questions_use_normal_search(question):
    assert parse_jira_question(question) is None


def test_named_person_count_and_limit():
    q = parse_jira_question("how many open bugs are assigned to Naresh Mandloi?")
    assert (q.person, q.role, q.status, q.issue_types, q.count_only) == (
        "Naresh Mandloi", "assignee", "open", ["Bug"], True)
    assert parse_jira_question("last 5 jira created by me").limit == 5


def issue(key, summary, reporter="Amit Kushwah"):
    return {"key": key, "fields": {"summary": summary, "status": {"name": "Resolved"},
            "reporter": {"displayName": reporter}, "assignee": None,
            "created": "2026-09-20T10:00:00.000+0000", "resolutiondate": "2026-09-25T10:00:00.000+0000"}}


def test_live_answer_runs_jql_and_formats_table(registry, monkeypatch):  # noqa: F811
    import src.sources.registry as registry_module
    import src.sources.credentials as credentials_module
    registry.create_jira_source(JiraSourceConfig(source_id="jira-1", base_url="https://jira.example.com",
                                                 project_key="WSQ", credential_id="c"), name="WSQ")
    monkeypatch.setattr(registry_module, "source_registry", registry)
    monkeypatch.setattr(credentials_module, "credential_store", FakeCredentials())
    session = Mock()
    session.get.return_value = response(200, {"issues": [issue("WSQ-2", "Fix | pipe"), issue("WSQ-1", "Login")],
                                              "total": 2})
    monkeypatch.setattr("src.sources.atlassian.requests.Session", lambda: session)

    answer, keys = answer_jira_question("all jira created by me last month", SourceFilter(["jira-1"], []))
    params = session.get.call_args.kwargs["params"]
    assert params["jql"] == 'project = "WSQ" AND reporter = currentUser() AND created >= -30d ORDER BY created DESC'
    assert session.get.call_args.args[0] == "https://jira.example.com/rest/api/2/search"
    assert answer.startswith("**2 issues** in WSQ reported by you (created in the last month)")
    assert "| 1 | [WSQ-2](https://jira.example.com/browse/WSQ-2) | Fix / pipe | Resolved | Amit Kushwah |" in answer
    assert "[Open this search in Jira](https://jira.example.com/issues/?jql=project%20%3D" in answer
    assert keys == ["WSQ-2", "WSQ-1"]
    # not a Jira source selected -> not answered here
    assert answer_jira_question("all jira created by me", SourceFilter(["bb-1"], [])) is None


def test_live_answer_reports_unknown_user_and_bad_jql(registry, monkeypatch):  # noqa: F811
    import src.sources.registry as registry_module
    import src.sources.credentials as credentials_module
    registry.create_jira_source(JiraSourceConfig(source_id="jira-1", base_url="https://jira.example.com",
                                                 project_key="WSQ", credential_id="c"), name="WSQ")
    monkeypatch.setattr(registry_module, "source_registry", registry)
    monkeypatch.setattr(credentials_module, "credential_store", FakeCredentials())
    session = Mock()
    session.get.return_value = response(200, [])
    monkeypatch.setattr("src.sources.atlassian.requests.Session", lambda: session)
    answer, _ = answer_jira_question("open bugs assigned to Nobody Here")
    assert "No Jira user matching **Nobody Here**" in answer

    session.get.return_value = response(400, {"errorMessages": ["Field 'resolved' does not exist"]})
    answer, _ = answer_jira_question("my resolved jira")
    assert "Jira rejected the query: Field 'resolved' does not exist" in answer


@pytest.mark.parametrize("question,person,role", [
    ("tell me jira Deversh jani", "Deversh jani", "any"),          # "tell me" is not "my"
    ("give me open jira of Naresh Mandloi", "Naresh Mandloi", "any"),
    ("Deversh's bugs", "Deversh", "any"),
    ("show me my open jira", "me", "any"),
    ("give me last month all jira which is create by me", "me", "reporter"),
    ("open bugs assigned to Naresh Mandloi in last month", "Naresh Mandloi", "assignee"),
])
def test_person_detection(question, person, role):
    query = parse_jira_question(question)
    assert (query.person, query.role) == (person, role)


def test_user_lookup_falls_back_to_first_name_and_matches_all_words():
    from unittest.mock import Mock
    from src.sources.jira_query import _resolve_user
    client = Mock()
    client.api.cloud = False
    client.api.json = lambda r, what: r.json()
    users = [{"name": "nxa111", "displayName": "Deversh Patel"}, {"name": "nxa222", "displayName": "Deversh Jani"}]
    client._get.side_effect = [response(200, []), response(200, users)]   # full name: none; first name: two
    assert _resolve_user(client, "Deversh jani") == '"nxa222"'
    assert client._get.call_args_list[1].args[1]["username"] == "Deversh"
    client._get.side_effect = [response(200, users)]
    assert _resolve_user(client, "Nobody Else") is None
