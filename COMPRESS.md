# SCRFD-500M parameter compression (no training)

Weight-space **COMPRESS** run after Hard-AP repair failed to transfer. Goal: cut
parameters while keeping WIDER Face AP within ±2.0 of baseline on DEV, then
confirm on holdout.

## Result (selected)

| | Baseline `det_500m.onnx` | **H44b** compressed |
|--|--:|--:|
| **Params** | 626,332 | **453,470 (−27.6%)** |
| **Conv GMACs** @640 | 0.734 | **0.581 (−20.7%)** |
| **ORT forward** (CPU) | ~7.7 ms | **~5.7 ms (~1.35×)** |

Artifact: [`models/compressed/det_500m_svd96_nokps.onnx`](models/compressed/det_500m_svd96_nokps.onnx)

### Accuracy

| Split | Easy | Medium | Hard | Δ Hard vs base |
|-------|-----:|-------:|-----:|---------------:|
| DEV (470) | 90.35 | 86.80 | 66.39 | −0.67 |
| CONFIRMATION (2756) | 90.22 | 88.01 | 67.88 | −0.71 |
| Full WIDER val (3226) | 90.24 | 87.84 | 67.67 | ~−0.7 |

Baseline DEV: 91.20 / 87.27 / 67.06 · CONFIRMATION: 90.65 / 88.37 / 68.59

## Method

1. **SVD low-rank PW factorization (H41b)**  
   The late MobileNet stage has five `288×288` pointwise convs (~66% of params).
   Each weight `W` is replaced by two `1×1` convs `B (r×288) → A (288×r)` with
   `A @ B ≈ SVD_r(W)`, **r = 96**, no fine-tuning.

2. **Strip KPS head (H44)**  
   Landmark outputs are unused for WIDER Easy/Medium/Hard AP (~35k params).
   Removing them does not change detection AP.

Reproduce:

```bash
conda activate basics
cd /home/prinshu/exp/weightSpace/scrfd
python wsp/svd_factor_pw.py --rank 96 --tag H41b_svd96
# then strip KPS (see H44b meta) or use the shipped ONNX above
python scripts/run_widerface_eval.py \
  --model models/compressed/det_500m_svd96_nokps.onnx --cpu
```

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

## Layout (after cleanup)

```
models/det_500m.onnx                 # original
models/compressed/det_500m_svd96_nokps.onnx
wsp/svd_factor_pw.py                  # SVD factor tool
wsp/structured_prune.py              # width-prune tool (negative result)
wsp/harness.py                       # DEV / CONFIRMATION eval
wsp/analysis/compress_outcome.json
wsp/analysis/H44b_test_results.json
wsp/analysis/H44b_latency_flops.json
```

## Notes

- Protocol: strict WSP (no gradients / training / distillation).
- Architecture change was **declared** for COMPRESS (not frozen-graph REPAIR).
- Official WIDER Face **test** GT is not public; holdout used here is the
  CONFIRMATION split plus full val.
- Full `detect()` wall time looks much faster on H44b partly because KPS decode
  is gone; prefer ORT forward + GMACs for apples-to-apples net cost.
