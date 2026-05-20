from __future__ import annotations

import re
from dataclasses import dataclass

import bcrypt

from db import (
    User,
    create_user,
    get_password_hash,
    get_user_by_email,
    get_user_by_id,
    row_to_user,
    set_last_login,
)


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class AuthResult:
    ok: bool
    user: "User | None" = None
    error: "str | None" = None


def is_valid_email(email: str) -> bool:
    return bool(EMAIL_RE.match(email.strip()))


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw, salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def register(
    conn,
    *,
    email: str,
    display_name: str,
    password: str,
    role: str = "educator",
    actor_user_id: int | None = None,
) -> AuthResult:
    email = email.strip()
    display_name = display_name.strip()

    if not email or not is_valid_email(email):
        return AuthResult(ok=False, error="Enter a valid email address.")
    if not display_name:
        return AuthResult(ok=False, error="Enter a display name.")
    if not password or len(password) < 8:
        return AuthResult(ok=False, error="Password must be at least 8 characters.")

    if get_user_by_email(conn, email) is not None:
        return AuthResult(ok=False, error="An account with that email already exists.")

    user_id = create_user(
        conn,
        email=email,
        display_name=display_name,
        password_hash=hash_password(password),
        role=role,
        actor_user_id=actor_user_id,
    )
    row = get_user_by_id(conn, user_id)
    return AuthResult(ok=True, user=row_to_user(row)) if row else AuthResult(ok=False, error="Failed to create user.")


def login(conn, *, email: str, password: str) -> AuthResult:
    email = email.strip()
    if not email or not password:
        return AuthResult(ok=False, error="Enter your email and password.")

    row = get_user_by_email(conn, email)
    if row is None:
        return AuthResult(ok=False, error="Invalid email or password.")

    user = row_to_user(row)
    if not user.is_active:
        return AuthResult(ok=False, error="Account is disabled. Contact an administrator.")

    password_hash = str(row["password_hash"])
    if not verify_password(password, password_hash):
        return AuthResult(ok=False, error="Invalid email or password.")

    set_last_login(conn, user_id=user.id)
    return AuthResult(ok=True, user=user)


def load_user(conn, *, user_id: int) -> User | None:
    row = get_user_by_id(conn, int(user_id))
    return row_to_user(row) if row else None


def get_user_password_hash(conn, *, user_id: int) -> str:
    return get_password_hash(conn, user_id=int(user_id))

