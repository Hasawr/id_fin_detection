import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).parent))

from fin_detector.batch_processor import BatchFINProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("benchmark")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Parallel Batch FIN Detection Performance")
    parser.add_argument("--image", type=str, default="images/back_1.png", help="Sample image path to test")
    parser.add_argument("--num-copies", type=int, default=100, help="Total number of images in batch (default: 100)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU and force CPU mode")
    parser.add_argument("--mode", type=str, choices=["mrz", "viz", "both"], default="mrz", help="Extraction mode")
    args = parser.parse_args()

    use_gpu = not args.no_gpu
    image_path = Path(args.image)
    if not image_path.exists():
        # Fallback to alternate sample image if default not found
        fallback = Path("images/sv arxa.jpg")
        if fallback.exists():
            image_path = fallback
        else:
            logger.error(f"Test image not found at {args.image} or fallback.")
            sys.exit(1)

    print("\n" + "="*70)
    print(f"🚀 STARTING BATCH INFERENCE BENCHMARK")
    print("="*70)
    print(f" • Test Image:        {image_path.name} ({image_path.stat().st_size / 1024:.1f} KB)")
    print(f" • Batch Size:        {args.num_copies} images")
    print(f" • Worker Pool Size:  {args.workers} processes")
    print(f" • Engine Mode:       {'GPU (CUDA Accelerated)' if use_gpu else 'CPU Native'}")
    print(f" • Target Extraction: {args.mode.upper()}")
    print("="*70 + "\n")

    # Read image bytes into memory to simulate real-world API request items
    raw_bytes = image_path.read_bytes()
    items = [
        {
            "id": f"img_{i+1:03d}_{image_path.name}",
            "input": raw_bytes
        }
        for i in range(args.num_copies)
    ]

    print(f"⏳ Initializing worker process pool with {args.workers} workers...")
    init_start = time.time()
    processor = BatchFINProcessor(max_workers=args.workers, use_gpu=use_gpu, debug=False)
    init_elapsed = time.time() - init_start
    print(f"✅ Worker pool ready in {init_elapsed:.2f} seconds.\n")

    print(f"⚡ Processing batch of {args.num_copies} images concurrently...")
    batch_start = time.time()
    results_summary = processor.process_batch(items, mode=args.mode)
    total_batch_time = time.time() - batch_start

    print("\n" + "="*70)
    print("📊 BENCHMARK PERFORMANCE RESULTS")
    print("="*70)
    print(f" • Total Batch Size:         {results_summary['total_images']} images")
    print(f" • Successful Extractions:   {results_summary['successful']}")
    print(f" • Failed Extractions:       {results_summary['failed']}")
    print(f" • Total Execution Time:     {results_summary['elapsed_seconds']:.4f} seconds")
    print(f" • Overall Throughput:       {results_summary['throughput_img_per_sec']:.2f} images/sec")
    print(f" • Average Per-Image Latency:{results_summary['avg_latency_ms']:.2f} ms")
    print("="*70)

    # Display sample result item
    if results_summary['results']:
        sample = results_summary['results'][0]
        print(f"\nSample Result Item [1/{args.num_copies}]:")
        print(f" • ID:             {sample.get('id')}")
        print(f" • FIN:            {sample.get('fin')}")
        print(f" • ID Number:      {sample.get('id_number')}")
        print(f" • Format:         {sample.get('card_format')}")
        conf = sample.get('confidence')
        conf_str = f"{conf:.4f}" if isinstance(conf, (int, float)) else "0.0000"
        print(f" • Confidence:     {conf_str}")
        print(f" • Latency:        {sample.get('elapsed_ms')} ms")
        print(f" • Checksum Valid: {sample.get('checksum_valid')}")

    print("\n🧹 Shutting down worker process pool...")
    processor.shutdown()
    print("✨ Benchmark completed successfully!\n")

if __name__ == "__main__":
    main()
