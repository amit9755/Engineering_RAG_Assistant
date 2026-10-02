"""Users and login sessions (SQLite, data/users.db).

Passwords: PBKDF2-HMAC-SHA256 with a random salt per user, never stored or logged
in plain text. Sessions: a random token in an HttpOnly cookie; only its SHA-256
hash is stored, so a copy of the database cannot be used to log in. Repeated
wrong passwords lock the account for a few minutes.
"""

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)

PBKDF2_ITERATIONS = 310_000
SESSION_SECONDS = 7 * 24 * 3600
MAX_FAILURES = 5
LOCK_SECONDS = 300
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")
MIN_PASSWORD = 8


@dataclass
class User:
    id: int
    username: str
    role: str
    must_change_password: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(check.hex(), digest)
    except (ValueError, AttributeError):
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class UserStore:
    def __init__(self, db_path: Path = Path("data/users.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._failures = {}   # username -> (count, first_failure_time)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    username              TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash         TEXT NOT NULL,
                    role                  TEXT NOT NULL DEFAULT 'user',
                    must_change_password  INTEGER NOT NULL DEFAULT 0,
                    created_at            REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash  TEXT PRIMARY KEY,
                    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    expires_at  REAL NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _user(row) -> User:
        return User(row["id"], row["username"], row["role"], bool(row["must_change_password"]))

    # ---------------------------------------------------------------- users

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def create_user(self, username: str, password: str, role: str = "user", must_change: bool = False) -> User:
        if not USERNAME.match(username or ""):
            raise ValueError("Username must be 3-40 letters, digits, '.', '_' or '-'")
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
        if role not in ("admin", "user"):
            raise ValueError("Role must be admin or user")
        with self._lock, self._connect() as conn:
            try:
                conn.execute("INSERT INTO users (username, password_hash, role, must_change_password, created_at) "
                             "VALUES (?, ?, ?, ?, ?)", (username, hash_password(password), role, int(must_change),
                                                        time.time()))
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"User {username} already exists") from exc
        logger.info("user_created", username=username, role=role)
        return self.get(username)

    def get(self, username: str) -> Optional[User]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return self._user(row) if row else None

    def list_users(self):
        with self._connect() as conn:
            return [self._user(r) for r in conn.execute("SELECT * FROM users ORDER BY id")]

    def set_password(self, username: str, password: str, must_change: bool = False, check_length: bool = True) -> None:
        if check_length and len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
        with self._lock, self._connect() as conn:
            updated = conn.execute("UPDATE users SET password_hash = ?, must_change_password = ? WHERE username = ?",
                                   (hash_password(password), int(must_change), username)).rowcount
            if not updated:
                raise ValueError(f"No user {username}")
            # Changing a password signs the user out everywhere else.
            conn.execute("DELETE FROM sessions WHERE user_id = (SELECT id FROM users WHERE username = ?)",
                         (username,))
        logger.info("password_changed", username=username)

    # ---------------------------------------------------------------- login

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """The user if the password is right; None otherwise. Raises PermissionError while locked."""
        key = (username or "").lower()
        count, first = self._failures.get(key, (0, 0))
        if count >= MAX_FAILURES and time.time() - first < LOCK_SECONDS:
            raise PermissionError("Too many failed attempts. Try again in a few minutes.")
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        # Always run the hash, so response time does not reveal whether the user exists.
        ok = verify_password(password or "", row["password_hash"] if row else hash_password("x" * 12))
        if not (row and ok):
            if time.time() - first >= LOCK_SECONDS:
                count, first = 0, time.time()
            self._failures[key] = (count + 1, first or time.time())
            logger.warning("login_failed", username=username)
            return None
        self._failures.pop(key, None)
        logger.info("login_succeeded", username=row["username"])
        return self._user(row)

    def create_session(self, user: User) -> str:
        token = secrets.token_urlsafe(32)
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                         (_token_hash(token), user.id, time.time() + SESSION_SECONDS))
        return token

    def user_for_session(self, token: str) -> Optional[User]:
        if not token:
            return None
        with self._connect() as conn:
            row = conn.execute("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
                               "WHERE s.token_hash = ? AND s.expires_at > ?",
                               (_token_hash(token), time.time())).fetchone()
        return self._user(row) if row else None

    def end_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token or ""),))

    # ---------------------------------------------------------------- first start

    def ensure_initial_admins(self, usernames=("admin1", "admin2"),
                              password_file: Path = Path("data/initial_admin_passwords.txt")) -> list:
        """
        Default admin accounts whose password is the same as the username (admin1 / admin1,
        admin2 / admin2), as requested for this local installation. Created on a fresh
        install; on an existing install, an admin still on a never-changed generated
        password is reset to the default. Passwords someone has changed are kept.
        """
        changed = []
        for name in usernames:
            user = self.get(name)
            if user is None:
                self._create_unchecked(name, name, role="admin")
                changed.append(name)
            elif user.must_change_password:
                self.set_password(name, name, must_change=False, check_length=False)
                changed.append(name)
        if password_file.exists():
            password_file.unlink()   # generated passwords from an earlier version are no longer valid
        if changed:
            logger.warning("default_admin_passwords_set", users=changed,
                           note="password equals username; change it with the key button if the app is shared")
        return changed

    def _create_unchecked(self, username: str, password: str, role: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO users (username, password_hash, role, must_change_password, created_at) "
                         "VALUES (?, ?, ?, 0, ?)", (username, hash_password(password), role, time.time()))
        logger.info("user_created", username=username, role=role)


user_store = UserStore()
