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


def test_audit_store_pii_is_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("API_KEYS", "k" * 32)
    monkeypatch.delenv("AUDIT_STORE_PII", raising=False)
    get_settings.cache_clear()
    assert get_settings().audit_store_pii is False

    monkeypatch.setenv("AUDIT_STORE_PII", "true")
    get_settings.cache_clear()
    assert get_settings().audit_store_pii is True
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


def test_configure_nvidia_libs_prepends_linux_ld_library_path(
    monkeypatch, tmp_path
) -> None:
    import services.id_fin.detector as detector
    from types import ModuleType

    cudnn_lib = tmp_path / "cudnn" / "lib"
    cublas_lib = tmp_path / "cublas" / "lib"
    nvrtc_lib = tmp_path / "nvrtc" / "lib"
    for path in (cudnn_lib, cublas_lib, nvrtc_lib):
        path.mkdir(parents=True)

    def fake_module(root: object) -> ModuleType:
        module = ModuleType("fake_nvidia")
        module.__path__ = [str(root)]  # type: ignore[attr-defined]
        return module

    roots = {
        "nvidia.cudnn": tmp_path / "cudnn",
        "nvidia.cublas": tmp_path / "cublas",
        "nvidia.cuda_nvrtc": tmp_path / "nvrtc",
    }

    monkeypatch.setattr(detector, "_NVIDIA_LIBS_CONFIGURED", False)
    monkeypatch.setattr(detector, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(detector, "_running_on_windows", lambda: False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/already/present")

    def fake_import(name: str):
        if name not in roots:
            raise ImportError(name)
        return fake_module(roots[name])

    monkeypatch.setattr(detector, "import_module", fake_import)

    detector.configure_nvidia_dll_directories()

    ld_path = os.environ["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(cudnn_lib.resolve()) in ld_path
    assert str(cublas_lib.resolve()) in ld_path
    assert str(nvrtc_lib.resolve()) in ld_path
    assert ld_path[-1] == "/already/present"
    # Idempotent: second call must not duplicate entries.
    before = os.environ["LD_LIBRARY_PATH"]
    detector.configure_nvidia_dll_directories()
    assert os.environ["LD_LIBRARY_PATH"] == before


def test_configure_nvidia_libs_warns_when_packages_missing(monkeypatch) -> None:
    import services.id_fin.detector as detector

    monkeypatch.setattr(detector, "_NVIDIA_LIBS_CONFIGURED", False)
    monkeypatch.setattr(detector, "_DLL_DIRECTORY_HANDLES", [])
    monkeypatch.setattr(detector, "_running_on_windows", lambda: False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    def missing_import(_name: str):
        raise ImportError("missing")

    monkeypatch.setattr(detector, "import_module", missing_import)
    detector.configure_nvidia_dll_directories()
    assert "LD_LIBRARY_PATH" not in os.environ

