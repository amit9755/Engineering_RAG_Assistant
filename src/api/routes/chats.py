"""Saved conversations: the most recent 10 chats, shown in the chat page's right panel."""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.chats import store

router = APIRouter(prefix="/chats", tags=["Chats"])


class ChatMessageIn(BaseModel):
    role: str
    content: str
    sources: List[str] = Field(default_factory=list)


class SaveChatRequest(BaseModel):
    messages: List[ChatMessageIn]


@router.get("", summary="List saved chats, newest first")
def list_chats():
    return store.chat_store.list()


@router.get("/{chat_id}", summary="Get a saved chat with its messages")
def get_chat(chat_id: str):
    chat = store.chat_store.get(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


@router.put("/{chat_id}", summary="Save (create or replace) a chat")
def save_chat(chat_id: str, request: SaveChatRequest):
    try:
        return store.chat_store.save(chat_id, [m.model_dump() for m in request.messages])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.delete("/{chat_id}", summary="Delete a saved chat")
def delete_chat(chat_id: str):
    if not store.chat_store.delete(chat_id):
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"message": "Chat deleted"}
