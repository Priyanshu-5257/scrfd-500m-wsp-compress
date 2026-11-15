# SCRFD-500M Weight-Space Compression

No-training compression of **SCRFD-500M** via weight-space programming (SVD
factorization + KPS strip). **−27.6% parameters**, **−20.7% GMACs**, holdout AP
within ~0.7 Hard of baseline.

Full write-up: **[COMPRESS.md](COMPRESS.md)** · License notes: [NOTICE](NOTICE)

## Results (H44b)

| | Params | GMACs@640 | CONFIRMATION E/M/H |
|--|--:|--:|--|
| Baseline | 626k | 0.73 | 90.65 / 88.37 / 68.59 |
| **Compressed** | **453k** | **0.58** | **90.22 / 88.01 / 67.88** |

Artifact: `models/compressed/det_500m_svd96_nokps.onnx`

## Quick start

```bash
# deps: onnxruntime, opencv, numpy, scipy, tqdm, pyyaml
pip install onnxruntime opencv-python-headless numpy scipy tqdm pyyaml onnx

# original
python scripts/run_widerface_eval.py --model models/det_500m.onnx --cpu

# compressed
python scripts/run_widerface_eval.py \
  --model models/compressed/det_500m_svd96_nokps.onnx --cpu
```

Place WIDER Face val images under `data/widerface/WIDER_val/images/` and official
GT mats under `data/widerface/eval_tools/eval_tools/ground_truth/` (see
[NOTICE](NOTICE); `data/` is not shipped).

## Layout

```
models/det_500m.onnx
models/compressed/det_500m_svd96_nokps.onnx   # selected compress artifact
eval/                                        # SCRFD ONNX + WIDER AP
scripts/run_widerface_eval.py
data/widerface/                              # val images + GT mats
wsp/                                         # harness, compress tools, splits
COMPRESS.md                                  # compress experiment write-up
```

## Compress summary

| | Params | GMACs@640 | CONFIRMATION E/M/H |
|--|--:|--:|--|
| Baseline | 626k | 0.73 | 90.65 / 88.37 / 68.59 |
| **H44b** | **453k** | **0.58** | **90.22 / 88.01 / 67.88** |

Method: SVD rank-96 factorization of late-backbone PW layers + strip KPS head  
(no training). Details, negatives, and latency: **[COMPRESS.md](COMPRESS.md)**.

## Published references (InsightFace)

| Model | Easy | Medium | Hard |
|---|---:|---:|---:|
| SCRFD_500M | 90.57 | 88.12 | 68.51 |
| SCRFD_500M_KPS | 90.97 | 88.44 | 69.49 |

Eval settings: letterbox **640×640**, score **0.02**, NMS **0.45**, IoU **0.5**.

## WSP harness

```bash
python wsp/harness.py baseline
python wsp/harness.py eval --hash <cand> --request-id rXXX --purpose ...
python wsp/harness.py confirm --hash <cand> --request-id rXXX --purpose ...
```

- DEV: `wsp/splits/dev.json` (470 images)
- CONFIRMATION: `wsp/splits/confirmation.json` (2756 images)
- Contract: `wsp/experiment.yaml`
