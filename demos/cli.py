import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.id_fin.detector import FINDetector
from services.id_fin.service import serialize_fin_detection


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a FIN from Azerbaijani ID card MRZ-side images."
    )
    parser.add_argument(
        "--mrz",
        type=Path,
        required=True,
        nargs="+",
        help="One or more MRZ-side images",
    )
    parser.add_argument("--cpu", action="store_true", help="Disable GPU acceleration")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", choices=["json", "text"], default="text")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    detector = FINDetector(
        use_gpu=not arguments.cpu,
        save_debug_images=arguments.debug,
    )
    results = [
        (image_path, detector.detect_from_mrz(image_path))
        for image_path in arguments.mrz
    ]

    if arguments.output == "json":
        payload = [
            {
                "file_name": image_path.name,
                **serialize_fin_detection(result),
            }
            for image_path, result in results
        ]
        print(json.dumps(payload, indent=2))
        return

    for image_path, result in results:
        print(f"{image_path.name}: {result.fin or 'Not found'}")
        print(f"Confidence: {result.confidence:.4f}")
        for note in result.notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
