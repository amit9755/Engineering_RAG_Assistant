"""Accounts, sign-in sessions, roles, first-start admins (temporary user database)."""

import pytest
from fastapi.testclient import TestClient

from src.auth.store import UserStore, hash_password, verify_password

pytestmark = pytest.mark.real_auth


@pytest.fixture
def users(tmp_path, monkeypatch):
    from src.auth import store as store_module
    users = UserStore(tmp_path / "users.db")
    monkeypatch.setattr(store_module, "user_store", users)
    # App startup would create admins (and a password file); tests create their own users.
    monkeypatch.setattr(users, "ensure_initial_admins", lambda *a, **k: [])
    return users


@pytest.fixture
def client(users):
    from src.api.main import app
    with TestClient(app) as client:
        yield client


def test_password_hashing():
    stored = hash_password("correct horse")
    assert stored.startswith("pbkdf2_sha256$") and "correct horse" not in stored
    assert verify_password("correct horse", stored) and not verify_password("wrong", stored)
    assert hash_password("same") != hash_password("same")  # random salt


def test_default_admins_use_username_as_password(users, tmp_path):
    create = UserStore.real_ensure_initial_admins   # the startup function (patched out elsewhere in tests)
    old_file = tmp_path / "initial.txt"
    old_file.write_text("admin1: generated\n")
    assert create(users, password_file=old_file) == ["admin1", "admin2"]
    assert not old_file.exists()
    for name in ("admin1", "admin2"):
        user = users.authenticate(name, name)
        assert user and user.is_admin and not user.must_change_password
    assert create(users, password_file=old_file) == []   # nothing to do on the next start


def test_existing_generated_passwords_are_reset_but_changed_ones_kept(users, tmp_path):
    create = UserStore.real_ensure_initial_admins
    users.create_user("admin1", "generated-pass-1", role="admin", must_change=True)   # never changed
    users.create_user("admin2", "generated-pass-2", role="admin", must_change=True)
    users.set_password("admin2", "my-own-password")                                 # admin2 changed it
    assert create(users, password_file=tmp_path / "x.txt") == ["admin1"]
    assert users.authenticate("admin1", "admin1")
    assert users.authenticate("admin2", "my-own-password") and not users.authenticate("admin2", "admin2")


def test_api_requires_sign_in(client):
    assert client.get("/api/v1/chats").status_code == 401
    assert client.get("/api/v1/sources/bitbucket").status_code == 401
    assert client.post("/api/v1/query", json={"question": "hi"}).status_code == 401
    assert client.get("/api/v1/health/ready").status_code == 200  # health stays public
    assert client.get("/api/v1/auth/me").status_code == 401


def test_sign_in_change_password_and_sign_out(client, users):
    users.create_user("admin1", "first-pass-123", role="admin", must_change=True)
    assert client.post("/api/v1/auth/login", json={"username": "admin1", "password": "nope"}).status_code == 401
    r = client.post("/api/v1/auth/login", json={"username": "admin1", "password": "first-pass-123"})
    assert r.status_code == 200 and r.json()["must_change_password"] is True
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert client.get("/api/v1/chats").status_code == 200

    assert client.post("/api/v1/auth/change-password", json={
        "current_password": "wrong", "new_password": "second-pass-456"}).status_code == 400
    assert client.post("/api/v1/auth/change-password", json={
        "current_password": "first-pass-123", "new_password": "short"}).status_code == 400
    r = client.post("/api/v1/auth/change-password", json={
        "current_password": "first-pass-123", "new_password": "second-pass-456"})
    assert r.status_code == 200 and r.json()["must_change_password"] is False
    assert client.get("/api/v1/auth/me").json()["username"] == "admin1"   # still signed in here

    client.post("/api/v1/auth/logout")
    assert client.get("/api/v1/chats").status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "admin1",
                                                  "password": "second-pass-456"}).status_code == 200


def test_chats_are_private_per_user(client, users, tmp_path, monkeypatch):
    from src.chats import store as chat_module
    from src.chats.store import ChatStore
    monkeypatch.setattr(chat_module, "chat_store", ChatStore(tmp_path / "chats.db"))
    users.create_user("admin1", "pass-admin-1", role="admin")
    users.create_user("admin2", "pass-admin-2", role="admin")
    chat = {"messages": [{"role": "user", "content": "admin1 secret question"}]}

    client.post("/api/v1/auth/login", json={"username": "admin1", "password": "pass-admin-1"})
    assert client.put("/api/v1/chats/sess-one", json=chat).status_code == 200
    client.post("/api/v1/auth/logout")

    client.post("/api/v1/auth/login", json={"username": "admin2", "password": "pass-admin-2"})
    assert client.get("/api/v1/chats").json() == []
    assert client.get("/api/v1/chats/sess-one").status_code == 404
    assert client.put("/api/v1/chats/sess-one", json=chat).status_code == 404   # cannot overwrite
    assert client.delete("/api/v1/chats/sess-one").status_code == 404
    client.post("/api/v1/auth/logout")

    client.post("/api/v1/auth/login", json={"username": "admin1", "password": "pass-admin-1"})
    assert [c["title"] for c in client.get("/api/v1/chats").json()] == ["admin1 secret question"]


def test_only_admins_change_sources(client, users):
    users.create_user("viewer", "viewer-pass-1", role="user")
    client.post("/api/v1/auth/login", json={"username": "viewer", "password": "viewer-pass-1"})
    assert client.get("/api/v1/sources/bitbucket").status_code == 200
    r = client.post("/api/v1/sources/confluence", json={"base_url": "https://c.example.com/display/X", "token": "t"})
    assert r.status_code == 403


def test_repeated_wrong_passwords_lock_the_account(users):
    users.create_user("admin1", "right-pass-1", role="admin")
    for _ in range(5):
        assert users.authenticate("admin1", "wrong") is None
    with pytest.raises(PermissionError):
        users.authenticate("admin1", "right-pass-1")
