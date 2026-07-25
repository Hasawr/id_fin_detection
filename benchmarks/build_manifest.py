import argparse
from getpass import getpass
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.id_fin.layout import TD1_LINE_LENGTH, TD2_LINE_LENGTH
from services.id_fin.validator import (
    is_valid_fin,
    is_valid_new_card_serial,
    is_valid_old_card_serial,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "fixtures.local.json"
SUPPORTED_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
EXPECTED_LINES = {
    "new_card": (3, TD1_LINE_LENGTH),
    "older_card": (2, TD2_LINE_LENGTH),
}


def hash_private_value(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def validate_output_path(output_path: Path) -> Path:
    resolved = output_path.resolve()
    if resolved != DEFAULT_OUTPUT.resolve():
        raise ValueError(
            "Ground truth must be written to the ignored "
            "benchmarks/fixtures.local.json path."
        )
    return resolved


def validate_mrz_lines(
    card_type: str,
    lines: list[str],
) -> list[str]:
    expected_count, expected_length = EXPECTED_LINES[card_type]
    normalized = [line.strip().upper() for line in lines]
    if len(normalized) != expected_count:
        raise ValueError(
            f"{card_type} requires {expected_count} MRZ lines."
        )
    for line in normalized:
        if len(line) != expected_length:
            raise ValueError(
                f"Every {card_type} MRZ line must contain "
                f"{expected_length} characters."
            )
        if re.fullmatch(r"[A-Z0-9<]+", line) is None:
            raise ValueError("MRZ lines may contain only A-Z, 0-9, and <.")
    return normalized


def discover_images(images_directory: Path) -> list[Path]:
    if not images_directory.is_dir():
        raise ValueError(f"Image directory does not exist: {images_directory}")
    images = sorted(
        path.resolve()
        for path in images_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise ValueError("No supported images were found.")
    unreadable = [path for path in images if cv2.imread(str(path)) is None]
    if unreadable:
        raise ValueError(f"Unreadable image: {unreadable[0]}")
    return images


def unique_case_id(image_path: Path, used_ids: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", image_path.stem.lower()).strip("-")
    base = base or "image"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def prompt_choice(prompt: str, choices: dict[str, str]) -> str:
    while True:
        value = input(prompt).strip().lower()
        if value in choices:
            return choices[value]
        print(f"Choose one of: {', '.join(choices)}")


def prompt_validated_secret(prompt: str, validator) -> str:
    while True:
        value = getpass(prompt).strip().upper()
        if validator(value):
            return value
        print("Value has an invalid format; try again.")


def build_case(image_path: Path, used_ids: set[str]) -> dict[str, object]:
    print(f"\nImage: {image_path.name}")
    card_type = prompt_choice(
        "Card type [n=new, o=older]: ",
        {"n": "new_card", "o": "older_card"},
    )
    fin = prompt_validated_secret("Expected FIN (hidden): ", is_valid_fin)
    serial_validator = (
        is_valid_new_card_serial
        if card_type == "new_card"
        else is_valid_old_card_serial
    )
    serial = ""
    if prompt_choice(
        "Add expected serial? [y/n]: ",
        {"y": "yes", "n": "no"},
    ) == "yes":
        serial = prompt_validated_secret(
            "Expected serial (hidden): ",
            serial_validator,
        )

    case: dict[str, object] = {
        "id": unique_case_id(image_path, used_ids),
        "path": image_path.as_posix(),
        "expected_card_type": card_type,
        "expected_fin_sha256": hash_private_value(fin),
    }
    if serial:
        case["expected_serial_sha256"] = hash_private_value(serial)

    if prompt_choice(
        "Add exact MRZ lines? [y/n]: ",
        {"y": "yes", "n": "no"},
    ) == "yes":
        line_count, _ = EXPECTED_LINES[card_type]
        lines = [
            input(f"MRZ line {index}: ")
            for index in range(1, line_count + 1)
        ]
        case["expected_mrz_lines"] = validate_mrz_lines(card_type, lines)
    return case


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an ignored, PII-safe OCR benchmark manifest."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = validate_output_path(arguments.output)
    images = discover_images(arguments.images.resolve())
    used_ids: set[str] = set()
    cases = [build_case(image, used_ids) for image in images]
    output_path.write_text(
        json.dumps({"cases": cases}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(cases)} cases to {output_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
