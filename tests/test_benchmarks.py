import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.benchmark_id_fin import (
    edit_distance,
    percentile,
    warm_latency_improvement,
)
from benchmarks.build_manifest import (
    discover_images,
    hash_private_value,
    unique_case_id,
    validate_mrz_lines,
    validate_output_path,
)


def test_native_decoder_dependencies_are_pinned_to_safe_versions() -> None:
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert "paddleocr==2.10.0" in requirements
    assert requirements.count("4.11.0.86") == 3
    assert "4.6.0.66" not in requirements


def test_benchmark_metric_helpers() -> None:
    assert percentile([0.4, 0.1, 0.2, 0.3], 0.5) == 0.3
    assert edit_distance("ABC<<<", "ABC<<") == 1


def test_warm_latency_gate_requires_20_percent_improvement() -> None:
    assert warm_latency_improvement(0.08, 0.10) >= 0.2
    assert warm_latency_improvement(0.081, 0.10) < 0.2


def test_manifest_builder_hashes_private_values() -> None:
    digest = hash_private_value("7ABC123")
    assert digest == hashlib.sha256(b"7ABC123").hexdigest()
    assert "7ABC123" not in digest


def test_manifest_builder_restricts_output_and_validates_mrz(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="fixtures.local.json"):
        validate_output_path(tmp_path / "manifest.json")
    lines = validate_mrz_lines(
        "new_card",
        ["I<AZE" + "<" * 25, "1" * 30, "DOE<<" + "<" * 25],
    )
    assert len(lines) == 3
    with pytest.raises(ValueError, match="30 characters"):
        validate_mrz_lines("new_card", ["short", "1" * 30, "2" * 30])


def test_manifest_builder_discovers_images_and_deduplicates_ids(
    tmp_path,
) -> None:
    image = np.full((20, 30, 3), 255, dtype=np.uint8)
    first = tmp_path / "Card One.png"
    second_directory = tmp_path / "nested"
    second_directory.mkdir()
    second = second_directory / "Card One.jpg"
    assert cv2.imwrite(str(first), image)
    assert cv2.imwrite(str(second), image)

    discovered = discover_images(tmp_path)
    used_ids: set[str] = set()
    identifiers = [unique_case_id(path, used_ids) for path in discovered]

    assert len(discovered) == 2
    assert identifiers == ["card-one", "card-one-2"]
