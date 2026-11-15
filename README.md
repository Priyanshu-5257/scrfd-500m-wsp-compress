# SCRFD-500M Weight-Space Compression

No-training compression research on **SCRFD-500M**. Structural edits are an
**extension** to strict fixed-architecture WSP (declared below).

## Validated conclusions (current)

| Edit | Params | GMACs@640 | DEV Hard Δ | ORT forward (Phase-1)* | Landmarks |
|------|-------:|----------:|-----------:|-----------------------:|:---------:|
| **A** Baseline | 626,332 | 0.734 | — | 8.12 ms [8.08, 8.16] | yes |
| **B** KPS-only | 591,710 (−5.5%) | 0.637 (−13.2%) | **0.00 pp** | 6.27 ms | no |
| **C** SVD-only (r=96) | 488,092 (−22.1%) | 0.678 (−7.5%) | **−0.66 pp** | 8.15 ms | yes |
| **D** Combined | 453,470 (−27.6%) | 0.581 (−20.7%) | **−0.66 pp** | 5.98 ms | no |

\*Phase-1 finalize: 96 real WIDER DEV images, `intra_op_num_threads=4`, ORT 1.24.4,
AMD Ryzen 7 5800H; bracket = bootstrap 95% CI on the median. See
[PHASE1_LATENCY_BENCHMARK.csv](PHASE1_LATENCY_BENCHMARK.csv).

**Isolation:** ~80% of the combined param cut is SVD; ~20% is KPS strip. All
measured Hard AP drop is from SVD. KPS strip does not change box AP.

**Latency:** SVD-only does **not** improve ORT forward vs baseline under this
protocol (8.15 ≈ 8.12 ms). Prior multi× e2e “speedups” were NMS/candidate-count
artifacts (especially on synthetic noise) — [LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md).

**Fairness:** B/D are detection-only; A/C retain landmarks. Do not compare their
e2e `detect()` times as equivalent products.

### Historical (superseded for latency claims)

| Source | Claim | Status |
|--------|-------|--------|
| Early COMPRESS note | ORT ~7.7→5.7 ms (~1.35×) for D vs A | **Superseded** |
| Ablation synthetic-blob ORT | A 6.26 / D 9.40 ms (default threads) | Historical; Phase-1 fixed-thread real-image numbers preferred |
| Ablation synthetic e2e | A 52 ms vs D 12 ms | Artifact of noise candidate counts |

CONFIRMATION AP (historical, A & D only): A 90.65/88.37/68.59 · D 90.22/88.01/67.88

### Structural follow-up (Phases 2–4) — negative for latency targets

Weight-algebra found no exact BN fusion, no near-duplicate PW filters, and no
near-identity blocks. Spectrum-motivated extra factors / tiny channel slice
(**S1–S4**, KPS retained) were built without training. **None** met all research
gates (≥20% params↓, ≥15% ORT↓, Hard drop ≤1 pp, KPS kept). Best prior
KPS-preserving compress remains **C** (params yes, ORT target no).
CONFIRMATION not consumed. Details:
[STRUCTURAL_ANALYSIS.md](STRUCTURAL_ANALYSIS.md),
[COMPRESSION_EXPERIMENTS.md](COMPRESSION_EXPERIMENTS.md),
[BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md).

## Documents

| Doc | Contents |
|-----|----------|
| [ABLATION_REPORT.md](ABLATION_REPORT.md) | Four-way A/B/C/D + contribution math |
| [LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md) | Why e2e SVD “speedup” was NMS |
| [PHASE1_LATENCY_BENCHMARK.csv](PHASE1_LATENCY_BENCHMARK.csv) | Larger-N latency + CI |
| [STRUCTURAL_ANALYSIS.md](STRUCTURAL_ANALYSIS.md) | Phase-2 weight-algebra findings |
| [COMPRESSION_EXPERIMENTS.md](COMPRESSION_EXPERIMENTS.md) | Phase-3 candidates + negatives |
| [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md) | Phase-4 head-to-head |
| [COMPRESS.md](COMPRESS.md) | Method history |
| [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md) | Protocol / limitations |
| [NOTICE](NOTICE) | License / data |

## Quick start

```bash
pip install onnxruntime opencv-python-headless numpy scipy tqdm pyyaml onnx

python scripts/run_widerface_eval.py --model models/det_500m.onnx --cpu
python scripts/run_widerface_eval.py \
  --model models/compressed/det_500m_svd96_nokps.onnx --cpu   # detection-only

python wsp/build_ablation_variants.py
python wsp/run_phase1_latency_benchmark.py
python wsp/structural_analysis.py
python wsp/build_structural_candidates.py
python wsp/run_structural_benchmark.py
```

WIDER val images: `data/widerface/WIDER_val/images/` · GT mats: see [NOTICE](NOTICE).

## WSP profile

- **No** training, gradients, fine-tuning, distillation, or output regression.
- **Ablation/SVD factor** on PW weights: weight-algebra + bounded DEV/CONFIRMATION eval.
- **Structural compression** (channel prune, block rewrite): declared **extension**
  to strict fixed-architecture WSP — see structural docs.
- DEV (470) vs CONFIRMATION (2756) remain separate; freeze before confirm.

## Published references (InsightFace)

| Model | Easy | Medium | Hard |
|---|---:|---:|---:|
| SCRFD_500M | 90.57 | 88.12 | 68.51 |
| SCRFD_500M_KPS | 90.97 | 88.44 | 69.49 |

Eval: letterbox **640×640**, score **0.02**, NMS **0.45**, IoU **0.5**.
