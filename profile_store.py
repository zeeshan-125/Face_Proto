"""Local SQLite storage for onboarding faceprints."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(os.getenv("FACEPRINT_DB_PATH", Path(__file__).parent / "faceprints.sqlite3"))


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_store() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS enrolled_profiles (
                profile_id TEXT PRIMARY KEY,
                faceprint_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def normalize_profile_id(profile_id: str) -> str:
    normalized = profile_id.strip()
    if not normalized or len(normalized) > 100:
        raise ValueError("Profile ID must be between 1 and 100 characters")
    return normalized


def get_profile(profile_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT profile_id, faceprint_json, created_at FROM enrolled_profiles WHERE profile_id = ?",
            (profile_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "profile_id": row["profile_id"],
        "faceprint": json.loads(row["faceprint_json"]),
        "created_at": row["created_at"],
    }


def create_profile(profile_id: str, faceprint: list[float]) -> dict:
    created_at = datetime.now(timezone.utc).isoformat()
    encoded = json.dumps(faceprint, separators=(",", ":"), allow_nan=False)
    with _connect() as connection:
        connection.execute(
            "INSERT INTO enrolled_profiles (profile_id, faceprint_json, created_at) VALUES (?, ?, ?)",
            (profile_id, encoded, created_at),
        )
    return {"profile_id": profile_id, "created_at": created_at}


initialize_store()
