# ID FIN OCR benchmark

The benchmark measures PaddleOCR initialization, cold and warm inference,
throughput, card-type accuracy, FIN accuracy, and full-image fallback rate.

Identity images and raw FIN values must not be committed. Copy
`fixtures.example.json` to the ignored `fixtures.local.json`, point each case
at a private local image, and store only the SHA-256 hash of its expected FIN.
An optional `[x, y, width, height]` crop can isolate a card from a screenshot.

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 3 `
  --output benchmarks\results.local.json
```

Compare an optimized run with a saved baseline:

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 3 `
  --baseline benchmarks\baseline.local.json `
  --min-improvement 0.20 `
  --output benchmarks\results.local.json
```

The command exits with code `2` when any expected FIN or card type changes,
and code `3` when the configured latency improvement is not reached.

## Measured result

Measured on Windows 11 with Python 3.12.10, PaddlePaddle GPU 2.6.2,
CUDA 11.8, cuDNN 8.6, and an NVIDIA GeForce RTX 4060 Laptop GPU. The private
fixture set contains one TD1 card and one TD2 card.

- Baseline: 170.9 ms warm median, 5.89 images/s, 50% fallback rate.
- Optimized: 57.6 ms warm median, 17.32 images/s, 0% fallback rate.
- Change: 66.3% lower warm median latency with both FIN and card-type results
  unchanged.

The retained production settings are a 1600-pixel pre-OCR input cap and
Paddle's 960-pixel detection-side limit. On the same fixtures, lower
detection limits were slower: 736 measured 107.3 ms and 640 measured
124.9 ms. These numbers describe this small local fixture set and hardware;
rerun the benchmark with a larger representative private set before sizing
production capacity.
