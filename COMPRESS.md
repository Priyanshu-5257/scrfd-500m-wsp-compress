# SCRFD-500M parameter compression (no training)

Weight-space **COMPRESS** run after Hard-AP repair failed to transfer. Goal: cut
parameters while keeping WIDER Face AP within ±2.0 of baseline on DEV, then
confirm on holdout.

**Corrected interpretation (ablation):** see
**[ABLATION_REPORT.md](ABLATION_REPORT.md)** and
**[EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)**. The shipped combined model mixes
two orthogonal edits; this document keeps the historical compress narrative and
marks claims that the four-way ablation revised.

## Result (selected combined = variant D)

| | Baseline `det_500m.onnx` (A) | **H44b** compressed (D) |
|--|--:|--:|
| **Params** | 626,332 | **453,470 (−27.6%)** |
| **Conv GMACs** @640 | 0.734 | **0.581 (−20.7%)** |
| **ORT forward** (CPU) | see ablation | **not a validated speedup** (see below) |

Artifact: [`models/compressed/det_500m_svd96_nokps.onnx`](models/compressed/det_500m_svd96_nokps.onnx)

### Accuracy

| Split | Easy | Medium | Hard | Δ Hard vs base |
|-------|-----:|-------:|-----:|---------------:|
| DEV (470) | 90.35 | 86.80 | 66.39 | −0.67 |
| CONFIRMATION (2756) | 90.22 | 88.01 | 67.88 | −0.71 |
| Full WIDER val (3226) | 90.24 | 87.84 | 67.67 | ~−0.7 |

Baseline DEV: 91.20 / 87.27 / 67.06 · CONFIRMATION: 90.65 / 88.37 / 68.59

Ablation DEV confirms A=B and C=D on box AP: **all Hard drop is from SVD**, not
KPS strip.

## Method

1. **SVD low-rank PW factorization (H41b / variant C)**  
   The late MobileNet stage has five `288×288` pointwise convs (~66% of params).
   Each weight `W` is replaced by two `1×1` convs `B (r×288) → A (288×r)` with
   `A @ B ≈ SVD_r(W)`, **r = 96**, no fine-tuning. Retains KPS outputs.

2. **Strip KPS head (H44 / variant B)**  
   Landmark outputs are unused for WIDER Easy/Medium/Hard AP (~35k params).
   Removing them does not change detection AP. It **does** remove three Conv
   heads (~13% GMACs) and changes the e2e pipeline (no landmark decode).

3. **Combined (H44b / variant D)** = (1) then (2).

Reproduce:

```bash
python wsp/build_ablation_variants.py   # A/B/C/D from same baseline
python wsp/run_ablation_benchmark.py
# or use the shipped ONNX:
python scripts/run_widerface_eval.py \
  --model models/compressed/det_500m_svd96_nokps.onnx --cpu
```

## Isolated contributions (measured)

| Transition | Params | GMACs | DEV Hard | Notes |
|------------|-------:|------:|---------:|-------|
| A→B KPS alone | −5.5% | −13.2% | 0.00 pp | Different product (no landmarks) |
| A→C SVD alone | −22.1% | −7.5% | −0.66 pp | Landmarks retained |
| A→D Combined | −27.6% | −20.7% | −0.66 pp | Params additive |

Full tables: [BENCHMARK_RESULTS.csv](BENCHMARK_RESULTS.csv),
[BENCHMARK_CONTRIBUTIONS.csv](BENCHMARK_CONTRIBUTIONS.csv).

## Latency — corrected stance

An earlier note in this file cited ~7.7 → ~5.7 ms ORT (~1.35×) for H44b vs
baseline. That figure is **superseded** by the controlled A–D ablation on the
same machine class:

| Variant | ORT median ms | Detect e2e median ms | Pipeline |
|---------|--------------:|---------------------:|----------|
| A | 6.26 | 51.63 | detect + landmarks |
| B | 5.13 | 43.82 | detect only |
| C | 6.55 | 13.20 | detect + landmarks |
| D | 9.40 | 12.00 | detect only |

On the original ablation timing inputs, **SVD did not show a reliable large ORT
`session.run` win**, and large e2e drops for C/D were composite. A follow-up
investigation ([LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md)) confirmed
the multi× e2e gap was mostly **fewer post-threshold boxes → cheaper NMS**,
especially on synthetic noise; with fixed threads and equalized candidate caps on
real WIDER images, that gap does not survive as a network-speed claim. Prefer
params, GMACs, and matched-pipeline ORT when claiming compute wins.

## What failed (honest negatives)

| Attempt | Idea | Outcome |
|---------|------|---------|
| H39 | Structured width prune 288→256/224/192 | Hard −4.5 to −18 |
| H40/H42 | Delete entire 288-ch DW+PW blocks | Hard ~−22 |
| H43 | Replace PW with Identity | Hard ~−23 |
| H41c | SVD rank 64 | Easy/Hard outside ±2 budget |

None of the four 288-ch mixers are near-identity; deleting any without
retraining collapses the detector. Low-rank **approximation** of the PW maps
is what preserved accuracy.

## Layout

```
models/det_500m.onnx
models/compressed/det_500m_svd96_nokps.onnx
models/ablation/{A_baseline,B_kps_only,C_svd_only,D_svd_nokps}.onnx
wsp/svd_factor_pw.py
wsp/build_ablation_variants.py
wsp/run_ablation_benchmark.py
wsp/structured_prune.py              # width-prune tool (negative result)
wsp/harness.py
wsp/analysis/ablation_*.json
ABLATION_REPORT.md · EXPERIMENT_AUDIT.md
```

## Notes

- Protocol: strict WSP (no gradients / training / distillation).
- Architecture change was **declared** for COMPRESS (not frozen-graph REPAIR).
- Official WIDER Face **test** GT is not public; holdout used here is the
  CONFIRMATION split plus full val.
- Landmark **accuracy** was never measured (no GT in workspace); only capability.
- B/C CONFIRMATION AP was **not** re-executed in the ablation; do not invent it.
