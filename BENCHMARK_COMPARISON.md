# Benchmark comparison — Phase 4

**Protocol:** `structural-benchmark-v1`  
**Hardware:** AMD Ryzen 7 5800H, onnxruntime 1.24.4, `CPUExecutionProvider`  
**Latency settings:** `intra_op_num_threads=4`, `inter_op=1`, input `1×3×640×640`, 48 real WIDER DEV images  
**AP:** DEV split (470 images), same harness thresholds as ablation  
**Raw:** `wsp/analysis/structural_benchmark.json`

Landmark **accuracy:** unavailable (no GT). Landmark **capability:** KPS outputs present iff `has_kps=yes`.

---

## Head-to-head

| Model | Params | Size (B) | GMACs | #Conv | ORT med ms | ORT p95 | e2e med ms | Pipeline | DEV E/M/H | KPS |
|-------|-------:|---------:|------:|------:|-----------:|--------:|-----------:|----------|----------:|:---:|
| **A** baseline | 626,332 | 2,524,817 | 0.734 | 60 | 8.30 | — | 21.97 | det+lm | 91.20 / 87.27 / 67.06 | yes |
| **C** SVD96 only | 488,092 | 1,972,636 | 0.678 | 65 | 8.21 | — | 21.78 | det+lm | 90.35 / 86.80 / 66.39 | yes |
| **D** SVD+noKPS | 453,470 | 1,832,884 | 0.581 | 62 | 6.65 | — | 17.46 | **det only** | 90.35 / 86.80 / 66.39 | **no** |
| S1_152_svd64 | 622,684 | — | 0.728 | 61 | 8.37 | — | 21.51 | det+lm | 87.06 / 85.01 / 63.98 | yes |
| S2_expand_svd70 | 613,356 | — | 0.728 | 61 | 8.66 | — | 20.40 | det+lm | 90.92 / 87.02 / 66.79 | yes |
| S3_multi_svd | 471,468 | — | 0.667 | 67 | 8.41 | — | 19.56 | det+lm | 86.34 / 84.18 / 62.33 | yes |
| S4_ch286 | 620,144 | — | 0.731 | 60 | 9.57 | — | 23.26 | det+lm | 90.96 / 87.41 / 66.95 | yes |

Phase-1 larger-N ORT medians (96 images) for A/B/C/D remain authoritative for the ablation set: see [PHASE1_LATENCY_BENCHMARK.csv](PHASE1_LATENCY_BENCHMARK.csv). This table’s A/C/D ORT values are a same-session remeasure for fair pairing with S*.

---

## vs A — research targets

Targets (aspirational, not assumed): ≥20% params↓, ≥15% ORT↓, Hard drop ≤1 pp, KPS preserved.

| Model | Params↓ % | ORT↓ % | Hard Δ pp | Gates passed |
|-------|----------:|-------:|----------:|--------------|
| C (prior) | 22.1 | ~1 | −0.66 | params+Hard+KPS; **not** ORT≥15% |
| D | 27.6 | ~20* | −0.66 | *ORT mixes KPS removal; **not** KPS-preserving |
| S1 | 0.6 | −0.9 | **−3.08** | fail |
| S2 | 2.1 | −4.3 | −0.27 | fail (compression/latency) |
| S3 | **24.7** | −1.3 | **−4.72** | fail (AP + latency) |
| S4 | 1.0 | −15.4 | −0.11 | fail (compression/latency) |

**CONFIRMATION:** not run — no candidate met all DEV gates for freeze.

---

## Interpretation

1. **Existing C** remains the best *KPS-preserving* param cut with Hard within ~0.7 pp, but it does **not** deliver a validated ≥15% ORT forward reduction (Phase-1: C≈A).
2. **D** looks faster on ORT largely because it drops KPS compute/outputs; it is a different product.
3. **Extra spectrum factors (S1/S3)** beyond PW288 r=96 hurt Hard AP more than the budget allows; stacking is not free.
4. **S2** (expand r=70) is relatively AP-safe but tiny and not faster.
5. **S4** (288→286) confirms tiny channel cuts can preserve AP but do not buy speed or meaningful size.

---

## Primary research question

> Can weight-space reasoning discover structurally redundant computations in SCRFD and remove them without training, producing a **genuine** reduction in measured inference latency?

**Answer (this phase): No — not under the stated success criteria.**

Weight-algebra found **spectral** compressibility in late PW layers (already used as SVD r=96) and weaker mid/expand spectra, but:

- Factorization **replaces one Conv with two**, which did not yield a reliable CPU ORT speedup here.
- Exact structural free lunch (BN fusion, duplicate-filter merge, near-identity blocks) was **absent** in the exported weights.
- Further no-training structural edits either **broke Hard AP** or **failed to improve ORT**.

This is reported as a **negative / inconclusive** result for latency-oriented structural WSP on SCRFD-500M, not as a weaker “winner.”
