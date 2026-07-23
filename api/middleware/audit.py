"""Record third-party API calls and persist request/response payloads locally."""

from __future__ import annotations

import json
import logging
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp, Message

from shared.audit import AuditStore, get_audit_store, sanitize_filename, service_from_path


logger = logging.getLogger(__name__)

SKIP_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
)


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        path = request.url.path
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in SKIP_PREFIXES):
            return await call_next(request)

        request_body = await request.body()
        request = Request(request.scope, self._replay_receive(request_body))

        started_at = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")

        content_type = response.headers.get("content-type", "")
        parsed_response = self._parse_json(response_body, content_type)
        error_code = self._extract_error_code(parsed_response, response_body, content_type)

        store = getattr(request.app.state, "audit_store", None) or get_audit_store()

        try:
            request_files, payload_dir = self._persist_request_payload(
                store=store,
                content_type=request.headers.get("content-type"),
                body=request_body,
            )
            store.record(
                method=request.method,
                path=path,
                service=service_from_path(path),
                status_code=response.status_code,
                latency_ms=latency_ms,
                api_key=request.headers.get("x-api-key"),
                error_code=error_code,
                client_host=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                request_content_type=request.headers.get("content-type"),
                request_query=request.url.query or None,
                request_files=request_files,
                response_body=parsed_response if parsed_response is not None else (
                    response_body.decode("utf-8", errors="replace") if response_body else None
                ),
                payload_dir=payload_dir,
            )
        except Exception:
            logger.exception("Failed to write API audit event for %s %s", request.method, path)

        headers = dict(response.headers)
        headers.pop("content-length", None)
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
            background=response.background,
        )

    @staticmethod
    def _replay_receive(body: bytes):
        sent = False

        async def receive() -> Message:
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        return receive

    def _persist_request_payload(
        self,
        *,
        store: AuditStore,
        content_type: str | None,
        body: bytes,
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not body:
            return [], None

        data_root = store.payload_dir.parent
        directory = store.allocate_payload_dir()
        files: list[dict[str, Any]] = []

        if content_type and "multipart/form-data" in content_type:
            files.extend(
                self._save_multipart_parts(
                    directory=directory,
                    data_root=data_root,
                    content_type=content_type,
                    body=body,
                )
            )
        else:
            extension = ".json" if content_type and "json" in content_type else ".bin"
            path = directory / f"request_body{extension}"
            path.write_bytes(body)
            files.append(
                {
                    "field": "body",
                    "file_name": path.name,
                    "content_type": content_type,
                    "size_bytes": len(body),
                    "saved_path": str(path.relative_to(data_root)).replace("\\", "/"),
                }
            )

        if not files:
            directory.rmdir()
            return [], None

        return files, str(directory.relative_to(data_root)).replace("\\", "/")

    def _save_multipart_parts(
        self,
        *,
        directory: Path,
        data_root: Path,
        content_type: str,
        body: bytes,
    ) -> list[dict[str, Any]]:
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + body
        )
        files: list[dict[str, Any]] = []
        part_index = 0

        for part in message.iter_parts():
            payload = part.get_payload(decode=True)
            if payload is None:
                continue

            field_name = part.get_param("name", header="content-disposition") or f"part_{part_index}"
            file_name = part.get_filename()
            part_content_type = part.get_content_type()

            if file_name:
                safe_name = sanitize_filename(file_name, fallback=f"{field_name}.bin")
                destination = directory / f"{part_index:02d}_{safe_name}"
                destination.write_bytes(payload)
                files.append(
                    {
                        "field": field_name,
                        "file_name": file_name,
                        "content_type": part_content_type,
                        "size_bytes": len(payload),
                        "saved_path": str(destination.relative_to(data_root)).replace("\\", "/"),
                    }
                )
            else:
                text_name = sanitize_filename(f"{field_name}.txt", fallback=f"field_{part_index}.txt")
                destination = directory / f"{part_index:02d}_{text_name}"
                destination.write_bytes(payload)
                files.append(
                    {
                        "field": field_name,
                        "file_name": None,
                        "content_type": part_content_type,
                        "size_bytes": len(payload),
                        "value_preview": payload.decode("utf-8", errors="replace")[:500],
                        "saved_path": str(destination.relative_to(data_root)).replace("\\", "/"),
                    }
                )
            part_index += 1

        return files

    @staticmethod
    def _parse_json(body: bytes, content_type: str) -> Any | None:
        if not body or "application/json" not in content_type:
            return None
        try:
            return json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _extract_error_code(
        parsed: Any,
        body: bytes,
        content_type: str,
    ) -> str | None:
        payload = parsed
        if payload is None and body and "application/json" in content_type:
            try:
                payload = json.loads(body)
            except (TypeError, ValueError, json.JSONDecodeError):
                return None

        if not isinstance(payload, dict):
            return None

        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code is not None else None

        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail[:120]
        return None
