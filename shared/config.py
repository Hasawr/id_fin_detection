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


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    use_gpu: bool
    debug: bool
    max_upload_bytes: int
    max_image_pixels: int
    max_batch_files: int
    audit_db_path: Path
    audit_payload_dir: Path


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
        debug=_as_bool(os.getenv("DEBUG")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
        max_image_pixels=int(os.getenv("MAX_IMAGE_PIXELS", "25000000")),
        max_batch_files=int(os.getenv("MAX_BATCH_FILES", "10")),
        audit_db_path=_resolve_path(
            os.getenv("AUDIT_DB_PATH", str(DEFAULT_AUDIT_DB_PATH)),
            DEFAULT_AUDIT_DB_PATH,
        ),
        audit_payload_dir=_resolve_path(
            os.getenv("AUDIT_PAYLOAD_DIR", str(DEFAULT_AUDIT_PAYLOAD_DIR)),
            DEFAULT_AUDIT_PAYLOAD_DIR,
        ),
    )
