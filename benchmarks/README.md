# ID FIN OCR benchmark

The benchmark measures PaddleOCR initialization, cold and warm inference,
per-attempt latency, throughput, card-type/FIN accuracy, optional serial
accuracy, OCR method counts, and peak sampled GPU memory.

Identity images and raw FIN values must not be committed. Copy
`fixtures.example.json` to the ignored `fixtures.local.json`, point each case
at a private local image, and store only SHA-256 hashes of its expected FIN
and optional card serial. An optional `[x, y, width, height]` crop can isolate
a card from a screenshot. Local manifests may include `expected_mrz_lines`
to calculate aggregate character-error and `<` filler-preservation rates.
Raw expected lines never appear in result JSON. To build the local manifest
without persisting raw FIN or serial values:

```powershell
python benchmarks\build_manifest.py --images C:\private\id-fin-corpus
```

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 3 `
  --output benchmarks\results.local.json
```

Add deterministic rotation, downscale, blur, and low-contrast variants:

```powershell
python benchmarks\benchmark_id_fin.py `
  --manifest benchmarks\fixtures.local.json `
  --runs 2 `
  --stress `
  --output benchmarks\results-stress.local.json
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
`method_counts` shows how often each TD1/TD2 image attempt was accepted,
including deskewed-strip and deskewed-full-image recovery.

## Current local quality check

Measured on Windows 11 with Python 3.12.10, PaddlePaddle GPU 2.6.2,
CUDA 11.8, cuDNN 8.9, and an NVIDIA GeForce RTX 4060 Laptop GPU. The available
fixture set contains one TD1 card and one TD2 card.

- First-valid PP-OCRv3 baseline: 50% exact FIN on these two cases.
- Quality-ranked PP-OCRv3: 100% exact FIN/card type; the difficult TD1 case
  selects the higher-quality full-image result instead of an earlier wrong
  binarized result.
- Stress check: 100% exact FIN/card type over the two originals plus eight
  deterministic degraded variants.
- Quality ranking costs additional attempts on ambiguous images. The measured
  warm median was 322 ms clean and 550 ms with stress variants; these tiny-set
  latency values are not production-capacity estimates.

The retained production settings are a 1600-pixel pre-OCR input cap and
Paddle's 960-pixel detection-side limit. On the same fixtures, lower
detection limits were slower: 736 measured 107.3 ms and 640 measured
124.9 ms. These numbers describe this small local fixture set and hardware;
run `build_manifest.py` against the representative 41-image folder before
using these measurements for production accuracy or capacity decisions.
