"""Record third-party API calls and persist request/response payloads locally."""

from __future__ import annotations

import json
import logging
import os
import time
from email.parser import BytesParser
from email.policy import default
from typing import Any, Callable

from starlette.datastructures import UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message

from shared.audit import AuditStore, get_audit_store, sanitize_filename, service_from_path
from shared.config import get_settings


logger = logging.getLogger(__name__)

SKIP_PREFIXES = (
    "/health",
    "/status",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/favicon.ico",
)


REQUEST_PARTS_STATE_KEY = "audit_request_parts"


def record_request_parts(
    request: Request,
    field: str,
    uploads: list[UploadFile],
) -> None:
    """Note what a caller uploaded, using FastAPI's already-parsed uploads.

    The middleware cannot describe the parts itself unless it buffers the
    whole request body, which is prohibitive for batches of up to 200 images.
    Routes call this instead: by the time they run, Starlette has already
    parsed the multipart stream (spooling large parts to disk), so reading
    the name, media type and size costs nothing extra.

    Only metadata is captured here. The image bytes are still retained solely
    when AUDIT_STORE_PAYLOADS is enabled.
    """
    parts: list[dict[str, Any]] = []
    for upload in uploads:
        # UploadFile.name is the spooled temp file's name, not the form
        # field, so the field is passed in by the route that declared it.
        size = getattr(upload, "size", None)
        if size is None:
            try:
                size = upload.file.seek(0, os.SEEK_END)
                upload.file.seek(0)
            except (AttributeError, OSError, ValueError):
                size = None
        parts.append(
            {
                "field": field,
                "file_name": upload.filename,
                "content_type": upload.content_type,
                "size_bytes": int(size) if size is not None else None,
            }
        )
    setattr(request.state, REQUEST_PARTS_STATE_KEY, parts)


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        path = request.url.path
        if any(path == prefix or path.startswith(f"{prefix}/") for prefix in SKIP_PREFIXES):
            return await call_next(request)

        settings = getattr(request.app.state, "settings", None)
        if settings is None:
            settings = get_settings()
        request_body = b""
        if settings.audit_store_payloads:
            request_body = await request.body()
            request = Request(
                request.scope,
                self._replay_receive(request_body),
            )

        started_at = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")

        content_type = response.headers.get("content-type", "")
        parsed_response = self._parse_json(response_body, content_type)
        error_code = self._extract_error_code(parsed_response)
        sanitized_response = self._sanitize_response(
            parsed_response,
            response.status_code,
            store_pii=settings.audit_store_pii,
        )

        store = getattr(request.app.state, "audit_store", None) or get_audit_store()

        try:
            # Nothing from a rejected request is retained, not even the part
            # metadata: an unauthenticated caller must not be able to write
            # attacker-chosen file names into the audit log.
            request_accepted = response.status_code < 400
            request_parts = (
                self._read_request_parts(
                    request.headers.get("content-type"),
                    request_body,
                )
                if request_accepted
                else []
            )
            request_files: list[dict[str, Any]] = [
                description for description, _payload in request_parts
            ]
            if request_accepted and not request_files:
                # Body was not buffered, so fall back to the metadata the
                # route recorded from its already-parsed uploads.
                request_files = list(
                    getattr(request.state, REQUEST_PARTS_STATE_KEY, None) or []
                )
            payload_dir = None
            if (
                settings.audit_store_payloads
                and request_accepted
                and request_body
                and store.can_store_payload(
                    len(request_body),
                    settings.audit_max_payload_bytes,
                )
            ):
                request_files, payload_dir = self._persist_request_payload(
                    store=store,
                    parts=request_parts,
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
                response_body=sanitized_response,
                payload_dir=payload_dir,
            )
            store.maybe_prune(
                retention_days=settings.audit_retention_days,
                max_payload_bytes=settings.audit_max_payload_bytes,
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

    def _read_request_parts(
        self,
        content_type: str | None,
        body: bytes,
    ) -> list[tuple[dict[str, Any], bytes]]:
        """Describe each uploaded part and keep its bytes for optional storage.

        The description (field, file name, media type, size) is recorded for
        every call, because it is what lets the audit answer "what did this
        integration send us?" without retaining the image itself. The bytes
        are only written to disk when AUDIT_STORE_PAYLOADS is enabled.
        """
        if not body:
            return []

        if content_type and "multipart/form-data" in content_type:
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + body
            )
            parts: list[tuple[dict[str, Any], bytes]] = []
            for index, part in enumerate(message.iter_parts()):
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                field_name = (
                    part.get_param("name", header="content-disposition")
                    or f"part_{index}"
                )
                file_name = part.get_filename()
                description: dict[str, Any] = {
                    "field": field_name,
                    "file_name": file_name,
                    "content_type": part.get_content_type(),
                    "size_bytes": len(payload),
                }
                if not file_name:
                    # Non-file form fields are small scalars, safe to keep.
                    description["value_preview"] = payload.decode(
                        "utf-8", errors="replace"
                    )[:500]
                parts.append((description, payload))
            return parts

        extension = ".json" if content_type and "json" in content_type else ".bin"
        return [
            (
                {
                    "field": "body",
                    "file_name": f"request_body{extension}",
                    "content_type": content_type,
                    "size_bytes": len(body),
                },
                body,
            )
        ]

    def _persist_request_payload(
        self,
        *,
        store: AuditStore,
        parts: list[tuple[dict[str, Any], bytes]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        if not parts:
            return [], None

        data_root = store.payload_dir.parent
        directory = store.allocate_payload_dir()
        files: list[dict[str, Any]] = []

        for index, (description, payload) in enumerate(parts):
            fallback = f"part_{index}.bin"
            safe_name = sanitize_filename(
                description.get("file_name") or f"{description['field']}.txt",
                fallback=fallback,
            )
            destination = directory / f"{index:02d}_{safe_name}"
            destination.write_bytes(payload)
            files.append(
                {
                    **description,
                    "saved_path": str(
                        destination.relative_to(data_root)
                    ).replace("\\", "/"),
                }
            )

        if not files:
            directory.rmdir()
            return [], None

        return files, str(directory.relative_to(data_root)).replace("\\", "/")

    @staticmethod
    def _parse_json(body: bytes, content_type: str) -> Any | None:
        if not body or "application/json" not in content_type:
            return None
        try:
            return json.loads(body)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    @classmethod
    def _extract_error_code(cls, payload: Any) -> str | None:
        if not isinstance(payload, dict):
            return None

        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            return str(code) if code is not None else None

        return cls._format_reject_detail(payload, status_code=None)

    @staticmethod
    def _format_reject_detail(
        payload: Any,
        *,
        status_code: int | None,
    ) -> str | None:
        """Keep a short, client-safe reject reason for the audit UI.

        FIN/serial/MRZ values are never present on these error payloads. The
        previous blanket "Request rejected." hid useful public messages such
        as missing API key, wrong path, or validation failures.
        """
        if not isinstance(payload, dict):
            return f"HTTP {status_code}" if status_code else None

        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if code and message:
                return f"{code}: {message}"[:160]
            if code is not None:
                return str(code)[:160]
            if message:
                return str(message)[:160]

        detail = payload.get("detail")
        if isinstance(detail, str) and detail.strip():
            return detail.strip()[:160]

        if isinstance(detail, list):
            parts: list[str] = []
            for item in detail[:3]:
                if isinstance(item, dict):
                    msg = item.get("msg") or item.get("message") or item.get("type")
                    loc = item.get("loc")
                    field = None
                    if isinstance(loc, (list, tuple)) and loc:
                        field = ".".join(
                            str(part) for part in loc if part != "body"
                        )
                    if msg and field:
                        parts.append(f"{field}: {msg}")
                    elif msg:
                        parts.append(str(msg))
                elif item is not None:
                    parts.append(str(item)[:80])
            if parts:
                return ("Validation: " + "; ".join(parts))[:160]

        return f"HTTP {status_code}" if status_code else None

    @classmethod
    def _sanitize_response(
        cls,
        payload: Any,
        status_code: int,
        *,
        store_pii: bool = False,
    ) -> Any:
        if not isinstance(payload, dict):
            return {"status_code": status_code}

        sanitized: dict[str, Any] = {
            "service": payload.get("service"),
            "version": payload.get("version"),
        }
        error = payload.get("error")
        if isinstance(error, dict):
            reason = cls._format_reject_detail(payload, status_code=status_code)
            sanitized["error"] = {
                "code": error.get("code"),
                "message": error.get("message"),
            }
            if reason:
                sanitized["detail"] = reason
            sanitized["status_code"] = status_code
            return sanitized
        if status_code >= 400:
            reason = cls._format_reject_detail(payload, status_code=status_code)
            sanitized["detail"] = reason or f"HTTP {status_code}"
            sanitized["status_code"] = status_code
            return sanitized

        data = payload.get("data")
        if not isinstance(data, dict):
            return sanitized
        if "fin" in data:
            sanitized["data"] = cls._sanitize_detection(
                data,
                store_pii=store_pii,
            )
            return sanitized

        results = data.get("results")
        if isinstance(results, list):
            sanitized_results = [
                {
                    "index": item.get("index"),
                    **cls._sanitize_detection(item, store_pii=store_pii),
                }
                for item in results
                if isinstance(item, dict)
            ]
            sanitized["data"] = {
                "count": len(sanitized_results),
                "results": sanitized_results,
            }
        return sanitized

    @staticmethod
    def _sanitize_detection(
        item: dict[str, Any],
        *,
        store_pii: bool = False,
    ) -> dict[str, Any]:
        details = item.get("mrz_details")
        safe_details = None
        if isinstance(details, dict):
            safe_details = {
                "card_type": details.get("card_type"),
                "method": details.get("method"),
                "checksum_valid": details.get("checksum_valid"),
                "card_serial_number": (
                    details.get("card_serial_number")
                    if store_pii
                    else "[REDACTED]"
                    if details.get("card_serial_number")
                    else None
                ),
            }
        return {
            "fin": (
                item.get("fin")
                if store_pii
                else "[REDACTED]"
                if item.get("fin")
                else None
            ),
            "confidence": item.get("confidence"),
            "mrz_details": safe_details,
        }
