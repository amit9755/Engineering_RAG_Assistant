"""Sign in / sign out / current user / change password."""

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel

from src.auth import store
from src.auth.deps import SESSION_COOKIE, require_user
from src.auth.store import SESSION_SECONDS, User

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


def _user_json(user: User) -> dict:
    return {"username": user.username, "role": user.role, "must_change_password": user.must_change_password}


def _set_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict",
                        secure=request.url.scheme == "https", path="/")


@router.post("/login", summary="Sign in")
def login(body: LoginRequest, request: Request, response: Response):
    try:
        user = store.user_store.authenticate(body.username.strip(), body.password)
    except PermissionError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    if not user:
        raise HTTPException(status_code=401, detail="Wrong username or password")
    _set_cookie(response, request, store.user_store.create_session(user))
    return _user_json(user)


@router.post("/logout", summary="Sign out")
def logout(request: Request, response: Response):
    store.user_store.end_session(request.cookies.get(SESSION_COOKIE, ""))
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"message": "Signed out"}


@router.get("/me", summary="The signed-in user")
def me(user: User = Depends(require_user)):
    return _user_json(user)


@router.post("/change-password", summary="Change your password")
def change_password(body: ChangePasswordRequest, request: Request, response: Response,
                    user: User = Depends(require_user)):
    if not store.user_store.authenticate(user.username, body.current_password):
        raise HTTPException(status_code=400, detail="Current password is wrong")
    if body.new_password == body.current_password:
        raise HTTPException(status_code=400, detail="Choose a different password")
    try:
        store.user_store.set_password(user.username, body.new_password)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # set_password signed out all sessions; start a fresh one for this browser.
    _set_cookie(response, request, store.user_store.create_session(store.user_store.get(user.username)))
    return _user_json(store.user_store.get(user.username))
