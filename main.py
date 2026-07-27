import sys
import argparse
import logging
import json
from pathlib import Path

# Add the project root to sys.path to run main.py directly
sys.path.append(str(Path(__file__).parent))

from fin_detector.detector import FINDetector
from fin_detector import FINDetectionOutput

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("main")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and extract FIN codes from Azerbaijani ID cards using PaddleOCR."
    )
    parser.add_argument("--viz", type=str, help="Path to VIZ-side (front) image")
    parser.add_argument("--mrz", type=str, help="Path to MRZ-side (back) image")
    parser.add_argument(
        "--mode", 
        choices=["viz", "mrz", "both"], 
        help="Which side(s) to process (default: inferred from provided paths)"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for PaddleOCR inference")
    parser.add_argument("--debug", action="store_true", help="Save annotated debug images to ./debug_output/")
    parser.add_argument(
        "--output", 
        choices=["json", "text"], 
        default="text", 
        help="Output format (default: text)"
    )
    return parser.parse_args()

def output_as_json(result: FINDetectionOutput):
    """Serialize the output to JSON format and print to stdout."""
    data = {
        "viz_fin": result.viz_fin,
        "mrz_fin": result.mrz_fin,
        "mrz_id_number": result.mrz_id_number,
        "viz_confidence": float(round(result.viz_confidence, 4)),
        "mrz_confidence": float(round(result.mrz_confidence, 4)),
        "viz_details": {
            "method": result.viz_result.method,
            "bbox": result.viz_result.bbox
        } if result.viz_result else None,
        "mrz_details": {
            "method": result.mrz_result.method,
            "id_number": result.mrz_result.id_number,
            "card_format": result.mrz_result.card_format,
            "line1": result.mrz_result.line1,
            "line2": result.mrz_result.line2,
            "line3": result.mrz_result.line3,
            "checksum_valid": result.mrz_result.checksum_valid
        } if result.mrz_result else None,
        "notes": result.notes
    }
    print(json.dumps(data, indent=2))

def output_as_text(result: FINDetectionOutput):
    """Print the output in human-readable text format."""
    print("=" * 60)
    print("        AZERBAIJANI ID CARD FIN CODE DETECTION RESULT        ")
    print("=" * 60)
    
    viz_fin_str = result.viz_fin if result.viz_fin else "Not Found / Not Scanned"
    viz_conf_str = f"({result.viz_confidence:.2f})" if result.viz_fin else ""
    print(f" VIZ Side (Front) FIN:  {viz_fin_str:<25} {viz_conf_str}")
    
    mrz_fin_str = result.mrz_fin if result.mrz_fin else "Not Found / Not Scanned"
    mrz_conf_str = f"({result.mrz_confidence:.2f})" if result.mrz_fin else ""
    print(f" MRZ Side (Back) FIN:   {mrz_fin_str:<25} {mrz_conf_str}")

    mrz_id_str = result.mrz_id_number if result.mrz_id_number else "Not Found / Not Scanned"
    print(f" MRZ Side ID Number:    {mrz_id_str:<25}")
    
    if result.mrz_result and result.mrz_result.line1:
        print("-" * 60)
        print(" MRZ Line Details:")
        print(f"   Card Format: {result.mrz_result.card_format}")
        print(f"   Line 1: {result.mrz_result.line1}")
        print(f"   Line 2: {result.mrz_result.line2}")
        print(f"   Line 3: {result.mrz_result.line3}")
        print(f"   Document Number Checksum Valid: {result.mrz_result.checksum_valid}")
        
    if result.notes:
        print("-" * 60)
        print(" Processing Notes:")
        for note in result.notes:
            print(f"   - {note}")
            
    print("=" * 60)

def main():
    args = parse_args()
    
    # Configure debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
        
    # Validate input paths
    if not args.viz and not args.mrz:
        logger.error("Error: You must provide at least one of --viz or --mrz image path.")
        sys.exit(1)
        
    # Infer mode if not explicitly set
    mode = args.mode
    if not mode:
        if args.viz and args.mrz:
            mode = "both"
        elif args.viz:
            mode = "viz"
        else:
            mode = "mrz"
            
    logger.info(f"Execution mode: {mode}")
    
    try:
        # Initialize detector
        detector = FINDetector(use_gpu=args.gpu, debug=args.debug)
        
        # Process based on mode
        if mode == "viz":
            result = detector.detect_from_viz(args.viz)
        elif mode == "mrz":
            result = detector.detect_from_mrz(args.mrz)
        else:  # both
            result = detector.detect_from_both(args.viz, args.mrz)
            
        # Display output
        if args.output == "json":
            output_as_json(result)
        else:
            output_as_text(result)
            
        sys.exit(0)
        
    except Exception as e:
        logger.critical(f"Fatal error during execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()