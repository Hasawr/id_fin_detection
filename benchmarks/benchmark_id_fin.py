import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.id_fin.detector import FINDetector, PRODUCTION_OCR_VERSION
from shared.config import get_settings


ATTEMPT_TIMING_PATTERN = re.compile(
    r"^OCR attempt ([a-z0-9_]+): ([0-9.]+)s\.$"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark ID FIN OCR without storing raw FIN values."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--min-improvement", type=float, default=0.20)
    parser.add_argument("--ocr-max-side", type=int, default=1600)
    # Default to the production setting so the benchmark measures what the
    # service actually runs, rather than a number that drifts away from it.
    parser.add_argument(
        "--det-limit-side-len",
        type=int,
        default=get_settings().ocr_det_limit_side_len,
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Add deterministic rotation, blur, scale, and contrast variants.",
    )
    return parser.parse_args()


def hash_fin(fin: str | None) -> str | None:
    if fin is None:
        return None
    return hashlib.sha256(fin.encode("ascii")).hexdigest()


def warm_latency_improvement(
    current_median: float,
    baseline_median: float,
) -> float:
    if baseline_median <= 0:
        raise ValueError("Baseline warm median must be positive.")
    return (baseline_median - current_median) / baseline_median


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile without values.")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def gpu_memory_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return max(
            int(line.strip())
            for line in completed.stdout.splitlines()
            if line.strip()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def edit_distance(expected: str, actual: str) -> int:
    previous = list(range(len(actual) + 1))
    for expected_index, expected_character in enumerate(expected, start=1):
        current = [expected_index]
        for actual_index, actual_character in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[actual_index] + 1,
                    previous[actual_index - 1]
                    + (expected_character != actual_character),
                )
            )
        previous = current
    return previous[-1]


def materialize_case(
    case: dict[str, Any],
    temporary_directory: Path,
) -> Path:
    source_path = Path(case["path"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Fixture does not exist: {source_path}")

    crop = case.get("crop")
    if crop is None:
        return source_path

    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Fixture is not a readable image: {source_path}")
    x, y, width, height = crop
    cropped = image[y : y + height, x : x + width]
    if cropped.size == 0:
        raise ValueError(f"Fixture crop is empty: {case['id']}")
    safe_case_name = hashlib.sha256(
        str(case["id"]).encode("utf-8")
    ).hexdigest()[:12]
    output_path = temporary_directory / f"{safe_case_name}.png"
    if not cv2.imwrite(str(output_path), cropped):
        raise OSError(f"Could not create benchmark crop: {case['id']}")
    return output_path


def materialize_stress_variants(
    case: dict[str, Any],
    source_path: Path,
    temporary_directory: Path,
) -> list[tuple[dict[str, Any], Path]]:
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Fixture is not readable: {case['id']}")
    height, width = image.shape[:2]
    rotation = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        3.0,
        1.0,
    )
    transformed = {
        "rotation_3deg": cv2.warpAffine(
            image,
            rotation,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        ),
        "downscale_50pct": cv2.resize(
            cv2.resize(
                image,
                (max(1, width // 2), max(1, height // 2)),
                interpolation=cv2.INTER_AREA,
            ),
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        ),
        "gaussian_blur": cv2.GaussianBlur(image, (5, 5), 1.2),
        "low_contrast": cv2.convertScaleAbs(image, alpha=0.65, beta=45),
    }
    variants: list[tuple[dict[str, Any], Path]] = []
    for variant_name, variant_image in transformed.items():
        variant_case = {
            **case,
            "id": f"{case['id']}::{variant_name}",
            "stress_variant": variant_name,
        }
        safe_name = hashlib.sha256(
            str(variant_case["id"]).encode("utf-8")
        ).hexdigest()[:12]
        output_path = temporary_directory / f"{safe_name}.png"
        if not cv2.imwrite(str(output_path), variant_image):
            raise OSError(f"Could not write stress variant: {variant_name}")
        variants.append((variant_case, output_path))
    return variants


def materialize_benchmark_cases(
    cases: list[dict[str, Any]],
    temporary_directory: Path,
    *,
    include_stress: bool,
) -> list[tuple[dict[str, Any], Path]]:
    materialized: list[tuple[dict[str, Any], Path]] = []
    for case in cases:
        fixture_path = materialize_case(case, temporary_directory)
        materialized.append((case, fixture_path))
        if include_stress:
            materialized.extend(
                materialize_stress_variants(
                    case,
                    fixture_path,
                    temporary_directory,
                )
            )
    return materialized


def benchmark() -> tuple[dict[str, Any], int]:
    arguments = parse_arguments()
    if arguments.runs < 2:
        raise ValueError("--runs must be at least 2 to measure warm latency.")

    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases:
        raise ValueError("Benchmark manifest contains no cases.")

    with tempfile.TemporaryDirectory(prefix="ocr-benchmark-") as directory:
        temporary_directory = Path(directory)
        materialized_cases = materialize_benchmark_cases(
            cases,
            temporary_directory,
            include_stress=arguments.stress,
        )
        benchmark_cases = [case for case, _ in materialized_cases]
        fixture_paths = [path for _, path in materialized_cases]

        initialization_started = time.perf_counter()
        detector = FINDetector(
            use_gpu=not arguments.cpu,
            max_ocr_side=arguments.ocr_max_side,
            det_limit_side_len=arguments.det_limit_side_len,
        )
        initialization_seconds = time.perf_counter() - initialization_started

        timings: list[float] = []
        warm_timings: list[float] = []
        timings_by_case: dict[str, list[float]] = {
            case["id"]: [] for case in benchmark_cases
        }
        attempt_timings: dict[str, list[float]] = {}
        gpu_memory_samples: list[int] = []
        initial_gpu_memory = (
            None if arguments.cpu else gpu_memory_used_mib()
        )
        if initial_gpu_memory is not None:
            gpu_memory_samples.append(initial_gpu_memory)
        mrz_character_edits = 0
        mrz_expected_characters = 0
        filler_expected = 0
        filler_preserved = 0
        case_results: list[dict[str, Any]] = []
        fallback_count = 0
        method_counts: Counter[str] = Counter()
        miss_categories: Counter[str] = Counter()
        attempts_to_result: list[int] = []
        checksum_valid_count = 0
        secondary_checksum_valid_count = 0
        case_accuracy = {
            case["id"]: {
                "fin_matches": True,
                "card_type_matches": True,
                **(
                    {"serial_matches": True}
                    if "expected_serial_sha256" in case
                    else {}
                ),
            }
            for case in benchmark_cases
        }
        total_inferences = 0

        for run_index in range(arguments.runs):
            for case, fixture_path in zip(
                benchmark_cases,
                fixture_paths,
                strict=True,
            ):
                started = time.perf_counter()
                result = detector.detect_from_mrz(fixture_path)
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                timings_by_case[case["id"]].append(elapsed)
                if run_index > 0:
                    warm_timings.append(elapsed)
                total_inferences += 1
                if not arguments.cpu:
                    memory_used = gpu_memory_used_mib()
                    if memory_used is not None:
                        gpu_memory_samples.append(memory_used)
                inference_attempts = 0
                attempted_method_names: set[str] = set()
                for note in result.notes:
                    timing_match = ATTEMPT_TIMING_PATTERN.match(note)
                    if timing_match:
                        inference_attempts += 1
                        attempted_method_names.add(timing_match.group(1))
                        attempt_timings.setdefault(
                            timing_match.group(1),
                            [],
                        ).append(float(timing_match.group(2)))
                attempts_to_result.append(inference_attempts)

                method = (
                    result.mrz_result.method
                    if result.mrz_result is not None
                    else "not_found"
                )
                method_counts[method] += 1
                did_fallback = bool(
                    attempted_method_names & FINDetector.RECOVERY_ATTEMPTS
                )
                if did_fallback:
                    fallback_count += 1
                card_type = (
                    result.mrz_result.card_type
                    if result.mrz_result is not None
                    else "unknown"
                )
                if result.mrz_result is not None:
                    checksum_valid_count += int(
                        result.mrz_result.checksum_valid
                    )
                    secondary_checksum_valid_count += int(
                        result.mrz_result.secondary_checksum_valid
                    )
                fin_matches = (
                    hash_fin(result.fin) == case["expected_fin_sha256"]
                )
                card_type_matches = (
                    card_type == case["expected_card_type"]
                )
                case_accuracy[case["id"]]["fin_matches"] &= fin_matches
                case_accuracy[case["id"]][
                    "card_type_matches"
                ] &= card_type_matches
                if "expected_serial_sha256" in case:
                    serial_number = (
                        result.mrz_result.card_serial_number
                        if result.mrz_result is not None
                        else None
                    )
                    case_accuracy[case["id"]]["serial_matches"] &= (
                        hash_fin(serial_number)
                        == case["expected_serial_sha256"]
                    )
                expected_mrz_lines = case.get("expected_mrz_lines")
                if (
                    isinstance(expected_mrz_lines, list)
                    and result.mrz_result is not None
                ):
                    actual_lines = [
                        result.mrz_result.line1,
                        result.mrz_result.line2,
                        result.mrz_result.line3,
                    ][: len(expected_mrz_lines)]
                    for expected_line, actual_line in zip(
                        expected_mrz_lines,
                        actual_lines,
                        strict=True,
                    ):
                        expected_text = str(expected_line)
                        mrz_character_edits += edit_distance(
                            expected_text,
                            actual_line,
                        )
                        mrz_expected_characters += len(expected_text)
                        for index, character in enumerate(expected_text):
                            if character != "<":
                                continue
                            filler_expected += 1
                            if (
                                index < len(actual_line)
                                and actual_line[index] == "<"
                            ):
                                filler_preserved += 1

                if run_index == arguments.runs - 1:
                    if result.fin is None:
                        miss_categories["null_fin"] += 1
                    elif not fin_matches:
                        miss_categories["wrong_fin"] += 1
                    if not card_type_matches:
                        miss_categories["wrong_card_type"] += 1
                    serial_matches = case_accuracy[case["id"]].get(
                        "serial_matches"
                    )
                    if serial_matches is False:
                        miss_categories["wrong_serial"] += 1
                    case_results.append(
                        {
                            "id": case["id"],
                            **case_accuracy[case["id"]],
                            "detected_card_type": card_type,
                            "method": method,
                            "attempts_run": inference_attempts,
                            "stress_variant": case.get("stress_variant"),
                            "checksum_valid": bool(
                                result.mrz_result
                                and result.mrz_result.checksum_valid
                            ),
                        }
                    )

    all_accurate = all(
        all(
            value
            for key, value in case.items()
            if key.endswith("_matches")
        )
        for case in case_results
    )
    warm_total = sum(warm_timings)
    metrics: dict[str, Any] = {
        "engine": "paddleocr-gpu" if not arguments.cpu else "paddleocr-cpu",
        "ocr_max_side": arguments.ocr_max_side,
        "det_limit_side_len": arguments.det_limit_side_len,
        "ocr_model_version": PRODUCTION_OCR_VERSION,
        "source_case_count": len(cases),
        "case_count": len(benchmark_cases),
        "stress_enabled": arguments.stress,
        "runs": arguments.runs,
        "initialization_seconds": round(initialization_seconds, 4),
        "cold_first_inference_seconds": round(timings[0], 4),
        "warm_median_seconds": round(statistics.median(warm_timings), 4),
        "warm_p95_seconds": round(percentile(warm_timings, 0.95), 4),
        "warm_images_per_second": round(len(warm_timings) / warm_total, 4),
        "cropped_pass_count": total_inferences - fallback_count,
        "full_image_fallback_count": fallback_count,
        "full_image_fallback_rate": round(
            fallback_count / total_inferences, 4
        ),
        "method_counts": dict(sorted(method_counts.items())),
        "miss_categories": dict(sorted(miss_categories.items())),
        "attempts_to_result": {
            "median": round(statistics.median(attempts_to_result), 2),
            "p95": percentile(
                [float(value) for value in attempts_to_result],
                0.95,
            ),
            "maximum": max(attempts_to_result),
        },
        "checksum_valid_rate": round(
            checksum_valid_count / total_inferences,
            4,
        ),
        "secondary_checksum_valid_rate": round(
            secondary_checksum_valid_count / total_inferences,
            4,
        ),
        "attempt_timings": {
            attempt_name: {
                "count": len(values),
                "median_seconds": round(statistics.median(values), 4),
                "p95_seconds": round(percentile(values, 0.95), 4),
                "total_seconds": round(sum(values), 4),
            }
            for attempt_name, values in sorted(attempt_timings.items())
        },
        "per_image_latency": {
            case_id: {
                "median_seconds": round(statistics.median(values), 4),
                "p95_seconds": round(percentile(values, 0.95), 4),
            }
            for case_id, values in timings_by_case.items()
        },
        "exact_match_rates": {
            key.removesuffix("_matches"): round(
                sum(bool(case.get(key)) for case in case_results)
                / len(case_results),
                4,
            )
            for key in (
                "fin_matches",
                "card_type_matches",
                "serial_matches",
            )
            if any(key in case for case in case_results)
        },
        "peak_gpu_memory_mib": (
            max(gpu_memory_samples) if gpu_memory_samples else None
        ),
        "all_expected_results_match": all_accurate,
        "cases": case_results,
    }
    if mrz_expected_characters:
        metrics["mrz_character_error_rate"] = round(
            mrz_character_edits / mrz_expected_characters,
            6,
        )
    if filler_expected:
        metrics["mrz_filler_preservation_rate"] = round(
            filler_preserved / filler_expected,
            6,
        )

    exit_code = 0 if all_accurate else 2
    if arguments.baseline is not None:
        baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
        baseline_median = float(baseline["warm_median_seconds"])
        improvement = warm_latency_improvement(
            metrics["warm_median_seconds"],
            baseline_median,
        )
        metrics["warm_median_improvement"] = round(improvement, 4)
        metrics["minimum_required_improvement"] = arguments.min_improvement
        if improvement < arguments.min_improvement and exit_code == 0:
            exit_code = 3

    rendered = json.dumps(metrics, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return metrics, exit_code


if __name__ == "__main__":
    _, benchmark_exit_code = benchmark()
    raise SystemExit(benchmark_exit_code)
