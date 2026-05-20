from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APP_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = APP_DIR / "data" / "app.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db_path() -> Path:
    env = os.environ.get("AI_DETECTOR_DB_PATH", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def exec_many(conn: sqlite3.Connection, sql: str, rows: Iterable[tuple[Any, ...]]) -> None:
    conn.executemany(sql, list(rows))
    conn.commit()


def exec_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def fetch_one(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    cur = conn.execute(sql, params)
    return cur.fetchone()


def fetch_all(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    cur = conn.execute(sql, params)
    return cur.fetchall()


def initialize(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('educator', 'super_admin')),
            is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0, 1)),
            created_at TEXT NOT NULL,
            last_login_at TEXT
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor_user_id INTEGER,
            action TEXT NOT NULL,
            target_user_id INTEGER,
            detail TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(actor_user_id) REFERENCES users(id),
            FOREIGN KEY(target_user_id) REFERENCES users(id)
        );
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            report_path TEXT,
            status TEXT NOT NULL CHECK(status IN ('uploaded', 'processed')),
            demo_scenario TEXT NOT NULL,
            label TEXT,
            confidence INTEGER,
            ai_probability INTEGER,
            word_count INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL,
            processed_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def audit(
    conn: sqlite3.Connection,
    *,
    actor_user_id: int | None,
    action: str,
    target_user_id: int | None = None,
    detail: str | None = None,
) -> None:
    exec_one(
        conn,
        "INSERT INTO audit_log (actor_user_id, action, target_user_id, detail, created_at) VALUES (?, ?, ?, ?, ?)",
        (actor_user_id, action, target_user_id, detail, utc_now_iso()),
    )


@dataclass(frozen=True)
class User:
    id: int
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: str
    last_login_at: str | None


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    user_id: int
    original_filename: str
    stored_path: str
    report_path: str | None
    status: str
    demo_scenario: str
    label: str | None
    confidence: int | None
    ai_probability: int | None
    word_count: int
    uploaded_at: str
    processed_at: str | None


def row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        role=str(row["role"]),
        is_active=bool(int(row["is_active"])),
        created_at=str(row["created_at"]),
        last_login_at=row["last_login_at"],
    )


def row_to_document(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        original_filename=str(row["original_filename"]),
        stored_path=str(row["stored_path"]),
        report_path=str(row["report_path"]) if row["report_path"] else None,
        status=str(row["status"]),
        demo_scenario=str(row["demo_scenario"]),
        label=str(row["label"]) if row["label"] else None,
        confidence=int(row["confidence"]) if row["confidence"] is not None else None,
        ai_probability=int(row["ai_probability"]) if row["ai_probability"] is not None else None,
        word_count=int(row["word_count"]),
        uploaded_at=str(row["uploaded_at"]),
        processed_at=str(row["processed_at"]) if row["processed_at"] else None,
    )


def count_users(conn: sqlite3.Connection) -> int:
    row = fetch_one(conn, "SELECT COUNT(1) AS n FROM users")
    return int(row["n"]) if row else 0


def get_user_by_email(conn: sqlite3.Connection, email: str) -> sqlite3.Row | None:
    return fetch_one(conn, "SELECT * FROM users WHERE lower(email) = lower(?)", (email.strip(),))


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    return fetch_one(conn, "SELECT * FROM users WHERE id = ?", (int(user_id),))


def list_users(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return fetch_all(
        conn,
        "SELECT * FROM users ORDER BY role DESC, is_active DESC, created_at DESC",
    )


def create_user(
    conn: sqlite3.Connection,
    *,
    email: str,
    display_name: str,
    password_hash: str,
    role: str,
    actor_user_id: int | None,
) -> int:
    cur = exec_one(
        conn,
        """
        INSERT INTO users (email, display_name, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, 1, ?)
        """,
        (email.strip(), display_name.strip(), password_hash, role, utc_now_iso()),
    )
    user_id = int(cur.lastrowid or 0)
    audit(conn, actor_user_id=actor_user_id, action="user_create", target_user_id=user_id, detail=f"role={role}")
    return user_id


def set_user_active(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    is_active: bool,
    actor_user_id: int | None,
) -> None:
    exec_one(conn, "UPDATE users SET is_active = ? WHERE id = ?", (1 if is_active else 0, int(user_id)))
    audit(
        conn,
        actor_user_id=actor_user_id,
        action="user_set_active",
        target_user_id=int(user_id),
        detail=f"is_active={1 if is_active else 0}",
    )


def set_user_role(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    role: str,
    actor_user_id: int | None,
) -> None:
    exec_one(conn, "UPDATE users SET role = ? WHERE id = ?", (role, int(user_id)))
    audit(conn, actor_user_id=actor_user_id, action="user_set_role", target_user_id=int(user_id), detail=f"role={role}")


def set_user_password_hash(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    password_hash: str,
    actor_user_id: int | None,
) -> None:
    exec_one(conn, "UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, int(user_id)))
    audit(conn, actor_user_id=actor_user_id, action="user_set_password", target_user_id=int(user_id))


def set_last_login(conn: sqlite3.Connection, *, user_id: int) -> None:
    exec_one(conn, "UPDATE users SET last_login_at = ? WHERE id = ?", (utc_now_iso(), int(user_id)))


def get_password_hash(conn: sqlite3.Connection, *, user_id: int) -> str:
    row = fetch_one(conn, "SELECT password_hash FROM users WHERE id = ?", (int(user_id),))
    return str(row["password_hash"]) if row else ""


def create_document(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    original_filename: str,
    stored_path: str,
    demo_scenario: str,
    word_count: int,
) -> int:
    cur = exec_one(
        conn,
        """
        INSERT INTO documents (
            user_id, original_filename, stored_path, status, demo_scenario, word_count, uploaded_at
        )
        VALUES (?, ?, ?, 'uploaded', ?, ?, ?)
        """,
        (int(user_id), original_filename, stored_path, demo_scenario, int(word_count), utc_now_iso()),
    )
    doc_id = int(cur.lastrowid or 0)
    audit(conn, actor_user_id=int(user_id), action="document_upload", detail=f"document_id={doc_id}")
    return doc_id


def list_documents_for_user(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    status: str | None = None,
) -> list[sqlite3.Row]:
    if status:
        return fetch_all(
            conn,
            """
            SELECT * FROM documents
            WHERE user_id = ? AND status = ?
            ORDER BY COALESCE(processed_at, uploaded_at) DESC, id DESC
            """,
            (int(user_id), status),
        )
    return fetch_all(
        conn,
        """
        SELECT * FROM documents
        WHERE user_id = ?
        ORDER BY COALESCE(processed_at, uploaded_at) DESC, id DESC
        """,
        (int(user_id),),
    )


def get_document_for_user(conn: sqlite3.Connection, *, user_id: int, document_id: int) -> sqlite3.Row | None:
    return fetch_one(
        conn,
        "SELECT * FROM documents WHERE id = ? AND user_id = ?",
        (int(document_id), int(user_id)),
    )


def mark_document_processed(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    document_id: int,
    report_path: str,
    label: str,
    confidence: int,
    ai_probability: int,
) -> None:
    exec_one(
        conn,
        """
        UPDATE documents
        SET status = 'processed', report_path = ?, label = ?, confidence = ?, ai_probability = ?, processed_at = ?
        WHERE id = ? AND user_id = ?
        """,
        (
            report_path,
            label,
            int(confidence),
            int(ai_probability),
            utc_now_iso(),
            int(document_id),
            int(user_id),
        ),
    )
    audit(conn, actor_user_id=int(user_id), action="document_process", detail=f"document_id={int(document_id)}")


def delete_documents_for_user(conn: sqlite3.Connection, *, user_id: int, document_ids: Iterable[int]) -> list[sqlite3.Row]:
    ids = [int(document_id) for document_id in document_ids]
    if not ids:
        return []

    placeholders = ",".join("?" for _ in ids)
    rows = fetch_all(
        conn,
        f"SELECT * FROM documents WHERE user_id = ? AND id IN ({placeholders})",
        (int(user_id), *ids),
    )
    if rows:
        exec_one(
            conn,
            f"DELETE FROM documents WHERE user_id = ? AND id IN ({placeholders})",
            (int(user_id), *[int(row["id"]) for row in rows]),
        )
        audit(
            conn,
            actor_user_id=int(user_id),
            action="document_delete",
            detail="document_ids=" + ",".join(str(row["id"]) for row in rows),
        )
    return rows
