"""FastAPI dependencies: the signed-in user from the session cookie."""

from fastapi import HTTPException, Request

from src.auth import store
from src.auth.store import User

SESSION_COOKIE = "rag_session"


def require_user(request: Request) -> User:
    user = store.user_store.user_for_session(request.cookies.get(SESSION_COOKIE, ""))
    if not user:
        raise HTTPException(status_code=401, detail="Please sign in")
    request.state.user = user
    return user


def require_user_admin_to_change(request: Request) -> User:
    """Any signed-in user may read; only admins may add, change or delete (sources, uploads)."""
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return require_user(request)
    return require_admin(request)


def require_admin(request: Request) -> User:
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Only admins can change knowledge sources")
    return user
