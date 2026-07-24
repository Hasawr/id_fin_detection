import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time
from typing import Any

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.id_fin.detector import FINDetector


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
    parser.add_argument("--det-limit-side-len", type=int, default=960)
    parser.add_argument("--cpu", action="store_true")
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
        fixture_paths = [
            materialize_case(case, temporary_directory) for case in cases
        ]

        initialization_started = time.perf_counter()
        detector = FINDetector(
            use_gpu=not arguments.cpu,
            max_ocr_side=arguments.ocr_max_side,
            det_limit_side_len=arguments.det_limit_side_len,
        )
        initialization_seconds = time.perf_counter() - initialization_started

        timings: list[float] = []
        warm_timings: list[float] = []
        case_results: list[dict[str, Any]] = []
        fallback_count = 0
        method_counts: Counter[str] = Counter()
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
            for case in cases
        }
        total_inferences = 0

        for run_index in range(arguments.runs):
            for case, fixture_path in zip(cases, fixture_paths, strict=True):
                started = time.perf_counter()
                result = detector.detect_from_mrz(fixture_path)
                elapsed = time.perf_counter() - started
                timings.append(elapsed)
                if run_index > 0:
                    warm_timings.append(elapsed)
                total_inferences += 1

                method = (
                    result.mrz_result.method
                    if result.mrz_result is not None
                    else "not_found"
                )
                method_counts[method] += 1
                did_fallback = not method.endswith(
                    ("mrz_strip", "mrz_roi")
                )
                if did_fallback:
                    fallback_count += 1
                card_type = (
                    result.mrz_result.card_type
                    if result.mrz_result is not None
                    else "unknown"
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

                if run_index == arguments.runs - 1:
                    case_results.append(
                        {
                            "id": case["id"],
                            **case_accuracy[case["id"]],
                            "detected_card_type": card_type,
                            "method": method,
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
        "case_count": len(cases),
        "runs": arguments.runs,
        "initialization_seconds": round(initialization_seconds, 4),
        "cold_first_inference_seconds": round(timings[0], 4),
        "warm_median_seconds": round(statistics.median(warm_timings), 4),
        "warm_p95_seconds": round(
            sorted(warm_timings)[
                min(len(warm_timings) - 1, int(len(warm_timings) * 0.95))
            ],
            4,
        ),
        "warm_images_per_second": round(len(warm_timings) / warm_total, 4),
        "cropped_pass_count": total_inferences - fallback_count,
        "full_image_fallback_count": fallback_count,
        "full_image_fallback_rate": round(
            fallback_count / total_inferences, 4
        ),
        "method_counts": dict(sorted(method_counts.items())),
        "all_expected_results_match": all_accurate,
        "cases": case_results,
    }

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
