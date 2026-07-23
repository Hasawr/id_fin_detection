"""SQLite-backed audit log for third-party API integration calls."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def fingerprint_api_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def label_api_key(api_key: str, index: int) -> str:
    suffix = api_key[-4:] if len(api_key) >= 4 else api_key
    return f"client-{index + 1} (…{suffix})"


def service_from_path(path: str) -> str | None:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) >= 2 and parts[0] == "v1":
        return parts[1]
    return None


def sanitize_filename(name: str, fallback: str = "upload.bin") -> str:
    cleaned = SAFE_NAME.sub("_", Path(name or fallback).name).strip("._")
    return cleaned[:120] or fallback


def summarize_response(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message")
        if code and message:
            return f"{code}: {message}"
        if code:
            return str(code)
        if message:
            return str(message)

    data = payload.get("data")
    if not isinstance(data, dict):
        detail = payload.get("detail")
        return str(detail)[:160] if detail is not None else None

    if "fin" in data:
        fin = data.get("fin")
        confidence = data.get("confidence")
        if fin:
            if isinstance(confidence, (int, float)):
                return f"FIN {fin} ({confidence:.2f})"
            return f"FIN {fin}"
        return "FIN not found"

    results = data.get("results")
    if isinstance(results, list):
        fins = [item.get("fin") for item in results if isinstance(item, dict) and item.get("fin")]
        count = data.get("count", len(results))
        if fins:
            preview = ", ".join(str(fin) for fin in fins[:3])
            extra = "" if len(fins) <= 3 else f" +{len(fins) - 3}"
            return f"batch {count}: {preview}{extra}"
        return f"batch {count}: no FIN found"

    return None


@dataclass(frozen=True)
class AuditEvent:
    id: int
    created_at: str
    method: str
    path: str
    service: str | None
    status_code: int
    success: bool
    latency_ms: float
    api_key_fingerprint: str | None
    error_code: str | None
    client_host: str | None
    user_agent: str | None
    request_content_type: str | None
    request_query: str | None
    request_files: list[dict[str, Any]]
    response_body: Any
    result_summary: str | None
    payload_dir: str | None


class AuditStore:
    def __init__(self, db_path: Path, payload_dir: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.payload_dir = Path(payload_dir) if payload_dir else self.db_path.parent / "audit_payloads"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.payload_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS api_audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    method TEXT NOT NULL,
                    path TEXT NOT NULL,
                    service TEXT,
                    status_code INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    api_key_fingerprint TEXT,
                    error_code TEXT,
                    client_host TEXT,
                    user_agent TEXT,
                    request_content_type TEXT,
                    request_query TEXT,
                    request_files_json TEXT,
                    response_body_json TEXT,
                    result_summary TEXT,
                    payload_dir TEXT
                )
                """
            )
            self._ensure_columns(connection)
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_api_audit_created_at
                ON api_audit_events (created_at DESC)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_api_audit_service
                ON api_audit_events (service)
                """
            )
            connection.commit()

    def _ensure_columns(self, connection: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(api_audit_events)").fetchall()
        }
        additions = {
            "user_agent": "TEXT",
            "request_content_type": "TEXT",
            "request_query": "TEXT",
            "request_files_json": "TEXT",
            "response_body_json": "TEXT",
            "result_summary": "TEXT",
            "payload_dir": "TEXT",
        }
        for column, column_type in additions.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE api_audit_events ADD COLUMN {column} {column_type}"
                )

    def record(
        self,
        *,
        method: str,
        path: str,
        service: str | None,
        status_code: int,
        latency_ms: float,
        api_key: str | None = None,
        error_code: str | None = None,
        client_host: str | None = None,
        user_agent: str | None = None,
        request_content_type: str | None = None,
        request_query: str | None = None,
        request_files: list[dict[str, Any]] | None = None,
        response_body: Any = None,
        payload_dir: str | None = None,
    ) -> int:
        created_at = datetime.now(timezone.utc).isoformat()
        success = 200 <= status_code < 400
        fingerprint = fingerprint_api_key(api_key)
        result_summary = summarize_response(response_body)
        files_json = json.dumps(request_files or [], ensure_ascii=False)
        response_json = (
            json.dumps(response_body, ensure_ascii=False, default=str)
            if response_body is not None
            else None
        )

        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO api_audit_events (
                    created_at,
                    method,
                    path,
                    service,
                    status_code,
                    success,
                    latency_ms,
                    api_key_fingerprint,
                    error_code,
                    client_host,
                    user_agent,
                    request_content_type,
                    request_query,
                    request_files_json,
                    response_body_json,
                    result_summary,
                    payload_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    method,
                    path,
                    service,
                    status_code,
                    int(success),
                    round(latency_ms, 2),
                    fingerprint,
                    error_code,
                    client_host,
                    user_agent,
                    request_content_type,
                    request_query,
                    files_json,
                    response_json,
                    result_summary,
                    payload_dir,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def allocate_payload_dir(self) -> Path:
        stamp = datetime.now(timezone.utc)
        directory = (
            self.payload_dir
            / stamp.strftime("%Y")
            / stamp.strftime("%m")
            / stamp.strftime("%d")
            / stamp.strftime("%H%M%S%f")
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def summary(self, *, hours: int | None = 24) -> dict[str, Any]:
        where, params = self._time_filter(hours)
        with self._lock, self._connect() as connection:
            totals = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS total,
                    COALESCE(SUM(success), 0) AS succeeded,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) AS failed,
                    COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
                FROM api_audit_events
                {where}
                """,
                params,
            ).fetchone()

            by_service = connection.execute(
                f"""
                SELECT
                    COALESCE(service, path) AS name,
                    COUNT(*) AS total,
                    COALESCE(SUM(success), 0) AS succeeded,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) AS failed
                FROM api_audit_events
                {where}
                GROUP BY COALESCE(service, path)
                ORDER BY total DESC
                """,
                params,
            ).fetchall()

            by_key = connection.execute(
                f"""
                SELECT
                    COALESCE(api_key_fingerprint, 'none') AS fingerprint,
                    COUNT(*) AS total,
                    COALESCE(SUM(success), 0) AS succeeded,
                    COALESCE(SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END), 0) AS failed
                FROM api_audit_events
                {where}
                GROUP BY COALESCE(api_key_fingerprint, 'none')
                ORDER BY total DESC
                """,
                params,
            ).fetchall()

        total = int(totals["total"])
        succeeded = int(totals["succeeded"])
        failed = int(totals["failed"])
        return {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "success_rate": (succeeded / total * 100.0) if total else 0.0,
            "avg_latency_ms": float(totals["avg_latency_ms"]),
            "by_service": [dict(row) for row in by_service],
            "by_key": [dict(row) for row in by_key],
        }

    def recent_events(
        self,
        *,
        limit: int = 100,
        hours: int | None = 24,
        service: str | None = None,
        success: bool | None = None,
    ) -> list[AuditEvent]:
        clauses: list[str] = []
        params: list[Any] = []

        if hours is not None:
            clauses.append("created_at >= ?")
            params.append(self._cutoff_iso(hours))
        if service:
            clauses.append("service = ?")
            params.append(service)
        if success is not None:
            clauses.append("success = ?")
            params.append(int(success))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM api_audit_events
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()

        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: int) -> AuditEvent | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM api_audit_events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_event(row) if row else None

    def _row_to_event(self, row: sqlite3.Row) -> AuditEvent:
        files_raw = row["request_files_json"] if "request_files_json" in row.keys() else None
        response_raw = row["response_body_json"] if "response_body_json" in row.keys() else None
        try:
            request_files = json.loads(files_raw) if files_raw else []
        except (TypeError, ValueError, json.JSONDecodeError):
            request_files = []
        try:
            response_body = json.loads(response_raw) if response_raw else None
        except (TypeError, ValueError, json.JSONDecodeError):
            response_body = response_raw

        return AuditEvent(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            method=str(row["method"]),
            path=str(row["path"]),
            service=row["service"],
            status_code=int(row["status_code"]),
            success=bool(row["success"]),
            latency_ms=float(row["latency_ms"]),
            api_key_fingerprint=row["api_key_fingerprint"],
            error_code=row["error_code"],
            client_host=row["client_host"],
            user_agent=row["user_agent"] if "user_agent" in row.keys() else None,
            request_content_type=(
                row["request_content_type"] if "request_content_type" in row.keys() else None
            ),
            request_query=row["request_query"] if "request_query" in row.keys() else None,
            request_files=request_files if isinstance(request_files, list) else [],
            response_body=response_body,
            result_summary=row["result_summary"] if "result_summary" in row.keys() else None,
            payload_dir=row["payload_dir"] if "payload_dir" in row.keys() else None,
        )

    @staticmethod
    def _cutoff_iso(hours: int) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return cutoff.isoformat()

    def _time_filter(self, hours: int | None) -> tuple[str, list[Any]]:
        if hours is None:
            return "", []
        return "WHERE created_at >= ?", [self._cutoff_iso(hours)]


def parse_query_string(query: str | None) -> dict[str, list[str]]:
    if not query:
        return {}
    return parse_qs(query, keep_blank_values=True)


_store: AuditStore | None = None
_store_lock = threading.Lock()


def get_audit_store(db_path: Path | None = None, payload_dir: Path | None = None) -> AuditStore:
    global _store
    with _store_lock:
        if _store is None:
            if db_path is None or payload_dir is None:
                from shared.config import get_settings

                settings = get_settings()
                db_path = db_path or settings.audit_db_path
                payload_dir = payload_dir or settings.audit_payload_dir
            _store = AuditStore(db_path, payload_dir=payload_dir)
        return _store


def reset_audit_store() -> None:
    global _store
    with _store_lock:
        _store = None
