# ============================================================
# src/sources/credentials.py - Secure credential storage
#
# Security design:
#   - Tokens are encrypted with Fernet (AES-128-CBC + HMAC-SHA256)
#   - Encryption key is generated once and stored in ./data/.credential_key
#   - The key file is gitignored and never committed
#   - Encrypted tokens are stored in SQLite as base64 strings
#   - GET APIs never return plaintext tokens
#   - Tokens are decrypted only inside integration clients (Bitbucket, Jira)
#     and immediately discarded after use
#
# Production upgrade path:
#   Replace LocalFernetCredentialStore with GCPSecretManagerCredentialStore
#   by swapping the implementation behind CredentialStore interface.
#   All callers use the same store() / retrieve() / delete() API.
# ============================================================

from __future__ import annotations

import base64
import os
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.observability.logger import get_logger

logger = get_logger(__name__)

# Default path for the SQLite credential DB and key file
_DATA_DIR = Path("./data")
_CREDENTIAL_DB = _DATA_DIR / "credentials.db"
_KEY_FILE = _DATA_DIR / ".credential_key"


# ============================================================
# Abstract interface - swap implementations without changing callers
# ============================================================

class CredentialStore(ABC):
    """Abstract credential store interface."""

    @abstractmethod
    def store(
        self,
        provider: str,
        label: str,
        plaintext_token: str,
        credential_id: Optional[str] = None,
    ) -> str:
        """Store a credential and return its credential_id."""
        ...

    @abstractmethod
    def retrieve(self, credential_id: str) -> str:
        """Return the plaintext token for a credential_id. Raises KeyError if not found."""
        ...

    @abstractmethod
    def update(self, credential_id: str, plaintext_token: str) -> None:
        """Replace the stored token for an existing credential."""
        ...

    @abstractmethod
    def delete(self, credential_id: str) -> None:
        """Delete a credential by ID."""
        ...

    @abstractmethod
    def exists(self, credential_id: str) -> bool:
        """Return True if the credential exists."""
        ...


# ============================================================
# Local Fernet implementation
# ============================================================

class LocalFernetCredentialStore(CredentialStore):
    """
    Credential store for local development.

    Tokens are encrypted with Fernet (symmetric AES) before storage
    in a local SQLite database. The encryption key is stored in a
    separate file (./data/.credential_key) that is gitignored.

    This is NOT intended for production. For production, replace
    this with GCPSecretManagerCredentialStore or similar.
    """

    def __init__(
        self,
        db_path: Path = _CREDENTIAL_DB,
        key_path: Path = _KEY_FILE,
    ):
        self._db_path = db_path
        self._key_path = key_path
        self._fernet = None
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_fernet()
        self._init_db()

    # ----------------------------------------------------------
    # Fernet key management
    # ----------------------------------------------------------

    def _init_fernet(self) -> None:
        """Load or generate the Fernet encryption key."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise RuntimeError(
                "cryptography package is required for credential storage. "
                "Install it with: pip install cryptography"
            )

        if self._key_path.exists():
            key = self._key_path.read_bytes()
            logger.info("credential_key_loaded", path=str(self._key_path))
        else:
            key = Fernet.generate_key()
            self._key_path.write_bytes(key)
            # Restrict permissions on Unix systems
            try:
                os.chmod(self._key_path, 0o600)
            except Exception:
                pass
            logger.info("credential_key_generated", path=str(self._key_path))

        self._fernet = Fernet(key)

    # ----------------------------------------------------------
    # SQLite schema
    # ----------------------------------------------------------

    def _init_db(self) -> None:
        """Create credentials table if it does not exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS credentials (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    label TEXT NOT NULL,
                    encrypted_value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
        logger.info("credential_db_initialized", path=str(self._db_path))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ----------------------------------------------------------
    # Encryption helpers
    # ----------------------------------------------------------

    def _encrypt(self, plaintext: str) -> str:
        """Encrypt a plaintext token and return base64-encoded ciphertext."""
        encrypted = self._fernet.encrypt(plaintext.encode("utf-8"))
        return base64.urlsafe_b64encode(encrypted).decode("ascii")

    def _decrypt(self, ciphertext_b64: str) -> str:
        """Decrypt a base64-encoded ciphertext and return the plaintext token."""
        encrypted = base64.urlsafe_b64decode(ciphertext_b64.encode("ascii"))
        return self._fernet.decrypt(encrypted).decode("utf-8")

    # ----------------------------------------------------------
    # CredentialStore interface implementation
    # ----------------------------------------------------------

    def store(
        self,
        provider: str,
        label: str,
        plaintext_token: str,
        credential_id: Optional[str] = None,
    ) -> str:
        """Encrypt and store a credential. Returns the credential_id."""
        cred_id = credential_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        encrypted = self._encrypt(plaintext_token)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO credentials (id, provider, label, encrypted_value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (cred_id, provider, label, encrypted, now, now),
            )

        logger.info("credential_stored", credential_id=cred_id, provider=provider)
        # Explicitly clear the plaintext from local scope
        del plaintext_token
        return cred_id

    def retrieve(self, credential_id: str) -> str:
        """Decrypt and return the plaintext token. Raises KeyError if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT encrypted_value FROM credentials WHERE id = ?",
                (credential_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Credential not found: {credential_id}")

        plaintext = self._decrypt(row["encrypted_value"])
        logger.info("credential_retrieved", credential_id=credential_id)
        return plaintext

    def update(self, credential_id: str, plaintext_token: str) -> None:
        """Replace the stored token for an existing credential."""
        if not self.exists(credential_id):
            raise KeyError(f"Credential not found: {credential_id}")

        encrypted = self._encrypt(plaintext_token)
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            conn.execute(
                "UPDATE credentials SET encrypted_value = ?, updated_at = ? WHERE id = ?",
                (encrypted, now, credential_id),
            )

        logger.info("credential_updated", credential_id=credential_id)
        del plaintext_token

    def delete(self, credential_id: str) -> None:
        """Delete a credential by ID."""
        with self._connect() as conn:
            conn.execute("DELETE FROM credentials WHERE id = ?", (credential_id,))
        logger.info("credential_deleted", credential_id=credential_id)

    def exists(self, credential_id: str) -> bool:
        """Return True if the credential exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM credentials WHERE id = ?",
                (credential_id,),
            ).fetchone()
        return row is not None


# ============================================================
# Singleton instance
# ============================================================

credential_store: CredentialStore = LocalFernetCredentialStore()
