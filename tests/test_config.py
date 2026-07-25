from dataclasses import replace

import pytest

from shared.config import Settings, validate_security_settings


def test_security_settings_reject_weak_and_duplicate_keys(tmp_path) -> None:
    base = Settings(
        api_keys=("a" * 32,),
        use_gpu=False,
        save_ocr_debug_images=False,
        max_upload_bytes=1024,
        max_image_pixels=100,
        max_batch_files=2,
        audit_db_path=tmp_path / "audit.db",
        audit_payload_dir=tmp_path / "payloads",
    )
    validate_security_settings(base)

    with pytest.raises(ValueError, match="at least 32"):
        validate_security_settings(replace(base, api_keys=("short",)))
    with pytest.raises(ValueError, match="duplicate"):
        validate_security_settings(
            replace(base, api_keys=("a" * 32, "a" * 32))
        )
