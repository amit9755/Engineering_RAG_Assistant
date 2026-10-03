"""Create Jira tickets from chat (any signed-in user): projects, issue types, fields, draft, create."""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.auth.deps import require_user
from src.auth.store import User
from src.sources.jobs import IndexingError

router = APIRouter(prefix="/jira", tags=["Jira tickets"])


class DraftRequest(BaseModel):
    question: str = Field(..., max_length=4000)
    conversation_history: List[Dict] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


class CreateIssueRequest(BaseModel):
    issue_type_id: str
    summary: str = Field(..., max_length=250)
    description: str = Field(default="", max_length=30000)
    labels: List[str] = Field(default_factory=list)
    fields: Dict[str, object] = Field(default_factory=dict)


async def _call(func, *args, **kwargs):
    try:
        return await run_in_threadpool(func, *args, **kwargs)
    except IndexingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except KeyError:
        raise HTTPException(status_code=422, detail="Saved Jira credentials are missing; re-add the Jira project.")


@router.get("/projects", summary="Jira projects tickets can be created in")
def projects():
    from src.sources.registry import source_registry
    return [{"source_id": s.id, "project_key": s.project_key, "base_url": s.base_url}
            for s in source_registry.list_jira_sources()]


@router.get("/{source_id}/issue-types", summary="Issue types of the project (Bug, Task, Story first)")
async def issue_types(source_id: str):
    from src.sources import jira_create
    return await _call(jira_create.issue_types, source_id)


@router.get("/{source_id}/fields", summary="Priority and other required fields for an issue type")
async def fields(source_id: str, issue_type_id: str):
    from src.sources import jira_create
    return await _call(jira_create.create_fields, source_id, issue_type_id)


@router.post("/draft", summary="Draft a ticket from the conversation (nothing is created)")
async def draft(request: DraftRequest):
    from src.sources import jira_create
    return await run_in_threadpool(jira_create.draft_ticket, request.question, request.conversation_history,
                                   request.sources)


@router.post("/{source_id}/issues", summary="Create the reviewed ticket in Jira")
async def create(source_id: str, request: CreateIssueRequest, user: User = Depends(require_user)):
    from src.sources import jira_create
    return await _call(jira_create.create_issue, source_id, request.issue_type_id, request.summary,
                       request.description, request.labels, request.fields, user.username)
