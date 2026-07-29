from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DB_PATH = PROJECT_ROOT / "data" / "audit.db"
DEFAULT_AUDIT_PAYLOAD_DIR = PROJECT_ROOT / "data" / "audit_payloads"


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _bounded_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}."
        )
    return value


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    use_gpu: bool
    save_ocr_debug_images: bool
    max_upload_bytes: int
    max_image_pixels: int
    max_batch_files: int
    audit_db_path: Path
    audit_payload_dir: Path
    max_batch_bytes: int = 50 * 1024 * 1024
    audit_store_payloads: bool = False
    audit_store_pii: bool = False
    audit_retention_days: int = 30
    audit_max_payload_bytes: int = 1024 * 1024 * 1024
    ocr_api_base_url: str = "http://127.0.0.1:8000"
    ocr_worker_count: int = 1
    ocr_concurrent_attempts: int = 1
    # Longest side PaddleOCR resizes to for text *detection*. Measured on the
    # benchmark fixtures: 736 reaches a valid MRZ in 22 attempts versus 32 at
    # 960, with identical FIN accuracy, because the smaller detection map
    # localizes the dense MRZ block more cleanly. Raise toward 960 only if
    # small-text recall regresses on your own images.
    ocr_det_limit_side_len: int = 736


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    api_keys = tuple(
        key.strip()
        for key in os.getenv("API_KEYS", "").split(",")
        if key.strip()
    )
    return Settings(
        api_keys=api_keys,
        use_gpu=_as_bool(os.getenv("USE_GPU"), default=True),
        save_ocr_debug_images=_as_bool(
            os.getenv("SAVE_OCR_DEBUG_IMAGES"),
        ),
        max_upload_bytes=_bounded_int(
            "MAX_UPLOAD_BYTES",
            10 * 1024 * 1024,
            minimum=1_024,
            maximum=100 * 1024 * 1024,
        ),
        max_image_pixels=_bounded_int(
            "MAX_IMAGE_PIXELS",
            25_000_000,
            minimum=1,
            maximum=100_000_000,
        ),
        max_batch_files=_bounded_int(
            "MAX_BATCH_FILES",
            100,
            minimum=1,
            maximum=1_000,
        ),
        audit_db_path=_resolve_path(
            os.getenv("AUDIT_DB_PATH", str(DEFAULT_AUDIT_DB_PATH)),
            DEFAULT_AUDIT_DB_PATH,
        ),
        audit_payload_dir=_resolve_path(
            os.getenv("AUDIT_PAYLOAD_DIR", str(DEFAULT_AUDIT_PAYLOAD_DIR)),
            DEFAULT_AUDIT_PAYLOAD_DIR,
        ),
        max_batch_bytes=_bounded_int(
            "MAX_BATCH_BYTES",
            50 * 1024 * 1024,
            minimum=1_024,
            maximum=1024 * 1024 * 1024,
        ),
        audit_store_payloads=_as_bool(
            os.getenv("AUDIT_STORE_PAYLOADS"),
        ),
        audit_store_pii=_as_bool(
            os.getenv("AUDIT_STORE_PII"),
        ),
        audit_retention_days=_bounded_int(
            "AUDIT_RETENTION_DAYS",
            30,
            minimum=1,
            maximum=3_650,
        ),
        audit_max_payload_bytes=_bounded_int(
            "AUDIT_MAX_PAYLOAD_BYTES",
            1024 * 1024 * 1024,
            minimum=0,
            maximum=1024 * 1024 * 1024 * 1024,
        ),
        ocr_api_base_url=(
            os.getenv("OCR_API_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/"),
        ocr_worker_count=_bounded_int(
            "OCR_WORKER_COUNT",
            1,
            minimum=1,
            maximum=8,
        ),
        ocr_concurrent_attempts=_bounded_int(
            "OCR_CONCURRENT_ATTEMPTS",
            1,
            minimum=1,
            maximum=4,
        ),
        ocr_det_limit_side_len=_bounded_int(
            "OCR_DET_LIMIT_SIDE_LEN",
            736,
            minimum=320,
            maximum=1920,
        ),
    )


def validate_security_settings(settings: Settings) -> None:
    if not settings.api_keys:
        raise ValueError("API_KEYS must contain at least one key.")
    if len(set(settings.api_keys)) != len(settings.api_keys):
        raise ValueError("API_KEYS must not contain duplicate keys.")
    for key in settings.api_keys:
        if key == "replace-with-a-long-random-key" or len(key) < 32:
            raise ValueError(
                "Every API key must be random and at least 32 characters."
            )
    if (
        settings.audit_store_payloads
        and settings.audit_max_payload_bytes <= 0
    ):
        raise ValueError(
            "AUDIT_MAX_PAYLOAD_BYTES must be positive when payload "
            "retention is enabled."
        )
