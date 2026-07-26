from dataclasses import replace
import os

import pytest

from shared.config import (
    Settings,
    get_settings,
    validate_security_settings,
)


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


def test_det_limit_side_len_is_read_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "k" * 32)
    monkeypatch.setenv("OCR_DET_LIMIT_SIDE_LEN", "960")
    get_settings.cache_clear()
    assert get_settings().ocr_det_limit_side_len == 960

    # Reject out-of-range values loudly rather than letting them reach
    # PaddleOCR, where a tiny detection map would silently wreck recall.
    monkeypatch.setenv("OCR_DET_LIMIT_SIDE_LEN", "16")
    get_settings.cache_clear()
    with pytest.raises(ValueError, match="OCR_DET_LIMIT_SIDE_LEN"):
        get_settings()
    get_settings.cache_clear()


def test_paddle_allocator_defaults_do_not_override_operator_settings(
    monkeypatch,
) -> None:
    from services.id_fin.detector import (
        PADDLE_ALLOCATOR_DEFAULTS,
        configure_paddle_allocator,
    )

    for flag in PADDLE_ALLOCATOR_DEFAULTS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv("FLAGS_allocator_strategy", "naive_best_fit")

    configure_paddle_allocator()

    # An explicit operator choice must survive.
    assert os.environ["FLAGS_allocator_strategy"] == "naive_best_fit"
    # Unset flags still receive our defaults.
    assert (
        os.environ["FLAGS_reallocate_gpu_memory_in_mb"]
        == PADDLE_ALLOCATOR_DEFAULTS["FLAGS_reallocate_gpu_memory_in_mb"]
    )
