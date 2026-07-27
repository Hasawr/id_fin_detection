import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Global worker detector reference for process pool initialization
_worker_detector = None

def _init_worker_process(use_gpu: bool, debug: bool):
    """
    Initializer function executed once per process worker startup.
    Pre-loads CUDA libraries and instantiates a singleton FINDetector instance.
    """
    global _worker_detector
    try:
        # Pre-load CUDA libraries in worker process
        from .ocr_engine import _preload_cuda_libraries
        _preload_cuda_libraries()
        
        from .detector import FINDetector
        _worker_detector = FINDetector(use_gpu=use_gpu, debug=debug)
        logger.info(f"Worker process PID {os.getpid()} initialized FINDetector (use_gpu={use_gpu}).")
    except Exception as e:
        logger.error(f"Worker process PID {os.getpid()} initialization failed: {e}", exc_info=True)
        raise e

def _worker_process_single(task: tuple) -> Dict[str, Any]:
    """
    Task handler running inside a worker process.
    Unpacks task: (item_id, item_input, mode)
    """
    global _worker_detector
    item_id, item_input, mode = task
    start_t = time.time()
    
    if _worker_detector is None:
        return {
            "id": item_id,
            "status": "error",
            "error_message": "Worker detector instance is uninitialized.",
            "elapsed_ms": round((time.time() - start_t) * 1000, 2)
        }
        
    try:
        if mode == "viz":
            output = _worker_detector.detect_from_viz(item_input)
        elif mode == "both" and isinstance(item_input, tuple):
            output = _worker_detector.detect_from_both(item_input[0], item_input[1])
        else:
            output = _worker_detector.detect_from_mrz(item_input)
            
        elapsed_ms = round((time.time() - start_t) * 1000, 2)
        
        # Format result dictionary
        result = {
            "id": item_id,
            "status": "success",
            "fin": output.mrz_fin or output.viz_fin,
            "id_number": output.mrz_id_number,
            "card_format": output.mrz_result.card_format if output.mrz_result else "UNKNOWN",
            "is_old_card": output.mrz_result.is_old_card if output.mrz_result else False,
            "checksum_valid": output.mrz_result.checksum_valid if output.mrz_result else False,
            "confidence": max(output.mrz_confidence, output.viz_confidence),
            "elapsed_ms": elapsed_ms,
            "notes": output.notes
        }
        return result
        
    except Exception as e:
        logger.error(f"Error processing item '{item_id}' in worker PID {os.getpid()}: {e}")
        return {
            "id": item_id,
            "status": "error",
            "error_message": str(e),
            "elapsed_ms": round((time.time() - start_t) * 1000, 2)
        }

class BatchFINProcessor:
    """
    High-performance parallel batch processor for FIN card detection.
    Manages a ProcessPoolExecutor with pre-initialized worker processes.
    """

    def __init__(self, max_workers: int = 4, use_gpu: bool = True, debug: bool = False):
        self.max_workers = max_workers
        self.use_gpu = use_gpu
        self.debug = debug
        self.executor = None
        self._start_pool()

    def _start_pool(self):
        """Start ProcessPoolExecutor with worker process initialization."""
        logger.info(f"Starting ProcessPoolExecutor with {self.max_workers} workers (use_gpu={self.use_gpu})...")
        self.executor = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_init_worker_process,
            initargs=(self.use_gpu, self.debug)
        )

    def process_batch(
        self, 
        items: List[Dict[str, Any]], 
        mode: str = "mrz"
    ) -> Dict[str, Any]:
        """
        Process a batch of images concurrently across process pool.
        
        `items`: List of dicts, e.g.:
            [
                {"id": "img1.jpg", "input": "/path/to/img1.jpg"},
                {"id": "img2.jpg", "input": b"raw_image_bytes..."}
            ]
        """
        if not items:
            return {
                "total_images": 0,
                "successful": 0,
                "failed": 0,
                "elapsed_seconds": 0.0,
                "throughput_img_per_sec": 0.0,
                "avg_latency_ms": 0.0,
                "mode": mode,
                "results": []
            }
            
        start_time = time.time()
        tasks = [(item["id"], item["input"], mode) for item in items]
        
        # Submit tasks to executor
        futures = [self.executor.submit(_worker_process_single, task) for task in tasks]
        
        results = []
        successful_count = 0
        failed_count = 0
        total_item_ms = 0.0
        
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                if res.get("status") == "success":
                    successful_count += 1
                else:
                    failed_count += 1
                total_item_ms += res.get("elapsed_ms", 0.0)
            except Exception as e:
                failed_count += 1
                results.append({
                    "id": "unknown",
                    "status": "error",
                    "error_message": f"Execution error: {str(e)}",
                    "elapsed_ms": 0.0
                })
                
        elapsed = time.time() - start_time
        throughput = len(items) / elapsed if elapsed > 0 else 0.0
        avg_latency = total_item_ms / len(items) if len(items) > 0 else 0.0
        
        # Sort results to maintain original item order if possible
        item_id_order = {item["id"]: i for i, item in enumerate(items)}
        results.sort(key=lambda r: item_id_order.get(r.get("id"), 999999))
        
        return {
            "total_images": len(items),
            "successful": successful_count,
            "failed": failed_count,
            "elapsed_seconds": round(elapsed, 4),
            "throughput_img_per_sec": round(throughput, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "mode": mode,
            "results": results
        }

    def shutdown(self):
        """Shut down process pool executor."""
        if self.executor:
            logger.info("Shutting down ProcessPoolExecutor...")
            self.executor.shutdown(wait=True)
            self.executor = None
