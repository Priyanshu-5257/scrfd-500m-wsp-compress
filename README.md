# SCRFD-500M Weight-Space Compression

No-training compression of **SCRFD-500M** via weight-space SVD (rank 96) plus
optional KPS-head strip. Combined artifact: **−27.6% parameters**,
**−20.7% GMACs**, DEV/CONFIRMATION Hard within ~0.7 of baseline.

**Ablation (verified):** most of the parameter cut is SVD (~80%); KPS strip is
~20% of params and **0 pp** of the Hard AP change. SVD alone did **not** show a
validated ORT forward speedup on the ablation CPU.

| Doc | Contents |
|-----|----------|
| **[ABLATION_REPORT.md](ABLATION_REPORT.md)** | Four-way A/B/C/D results + contribution math |
| **[LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md)** | Why e2e “SVD speedup” was NMS/candidate artifact |
| **[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)** | Protocol, hashes, limitations |
| **[BENCHMARK_RESULTS.csv](BENCHMARK_RESULTS.csv)** | Machine-readable metrics |
| **[COMPRESS.md](COMPRESS.md)** | Method history + negatives |
| [NOTICE](NOTICE) | License / data notes |

## Results

### Combined artifact (H44b = variant D)

| | Params | GMACs@640 | CONFIRMATION E/M/H |
|--|--:|--:|--|
| Baseline (A) | 626k | 0.73 | 90.65 / 88.37 / 68.59 |
| **Compressed (D)** | **453k** | **0.58** | **90.22 / 88.01 / 67.88** |

Artifact: `models/compressed/det_500m_svd96_nokps.onnx`  
(D is detection-only; baseline retains landmarks.)

### Ablation snapshot (DEV AP, same harness)

| Variant | Params | GMACs | ORT med ms* | DEV E/M/H | Landmarks |
|---------|-------:|------:|------------:|----------:|:---------:|
| A Baseline | 626k | 0.73 | 6.3 | 91.20 / 87.27 / 67.06 | yes |
| B KPS-only | 592k | 0.64 | 5.1 | 91.20 / 87.27 / 67.06 | no |
| C SVD-only | 488k | 0.68 | 6.6 | 90.35 / 86.80 / 66.39 | yes |
| D Combined | 453k | 0.58 | 9.4 | 90.35 / 86.80 / 66.39 | no |

\*AMD Ryzen 7 5800H, ORT 1.24.4, `session.run` only @640². See ablation report
for e2e `detect()` times and why they are not interchangeable across KPS vs no-KPS.

**Isolated contributions (params / Hard pp):** A→B −5.5% / 0.00 · A→C −22.1% / −0.66 · A→D −27.6% / −0.66

## Quick start

```bash
# deps: onnxruntime, opencv, numpy, scipy, tqdm, pyyaml, onnx
pip install onnxruntime opencv-python-headless numpy scipy tqdm pyyaml onnx

# original (landmarks)
python scripts/run_widerface_eval.py --model models/det_500m.onnx --cpu

# compressed (detection-only)
python scripts/run_widerface_eval.py \
  --model models/compressed/det_500m_svd96_nokps.onnx --cpu

# rebuild + measure A/B/C/D ablation
python wsp/build_ablation_variants.py
python wsp/run_ablation_benchmark.py
```

Place WIDER Face val images under `data/widerface/WIDER_val/images/` and official
GT mats under `data/widerface/eval_tools/eval_tools/ground_truth/` (see
[NOTICE](NOTICE); `data/` is not shipped).

## Layout

```
models/det_500m.onnx
models/compressed/det_500m_svd96_nokps.onnx   # D / H44b
models/ablation/{A,B,C,D}_*.onnx             # four-way variants
eval/                                        # SCRFD ONNX + WIDER AP
scripts/run_widerface_eval.py
wsp/                                         # harness, SVD/KPS tools, ablation
ABLATION_REPORT.md · EXPERIMENT_AUDIT.md
BENCHMARK_RESULTS.csv · BENCHMARK_CONTRIBUTIONS.csv
COMPRESS.md
```

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
