"""Saved conversations: the most recent 10 chats, shown in the chat page's right panel."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.auth.deps import require_user
from src.auth.store import User
from src.chats import store

router = APIRouter(prefix="/chats", tags=["Chats"])


class ChatMessageIn(BaseModel):
    role: str
    content: str
    sources: List[str] = Field(default_factory=list)


class SaveChatRequest(BaseModel):
    messages: List[ChatMessageIn]


@router.get("", summary="List saved chats, newest first")
def list_chats(user: User = Depends(require_user)):
    return store.chat_store.list(user.username)


@router.get("/{chat_id}", summary="Get a saved chat with its messages")
def get_chat(chat_id: str, user: User = Depends(require_user)):
    chat = store.chat_store.get(chat_id, user.username)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.put("/{chat_id}", summary="Save (create or replace) a chat")
def save_chat(chat_id: str, request: SaveChatRequest, user: User = Depends(require_user)):
    try:
        return store.chat_store.save(chat_id, [m.model_dump() for m in request.messages], user.username)
    except PermissionError:
        raise HTTPException(status_code=404, detail="Chat not found")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.delete("/{chat_id}", summary="Delete a saved chat")
def delete_chat(chat_id: str, user: User = Depends(require_user)):
    if not store.chat_store.delete(chat_id, user.username):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}
