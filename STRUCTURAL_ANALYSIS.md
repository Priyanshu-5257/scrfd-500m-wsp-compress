# Structural analysis — SCRFD-500M (Phase 2)

**Protocol:** `structural-analysis-v1`  
**Observation profile:** `weight-algebra` (no image execution during analysis)  
**Checkpoint:** `models/det_500m.onnx` (626,332 params)  
**Classification:** Structural compression (channel prune / block rewrite / op fusion) is an **extension** to strict fixed-architecture WSP. Declared here explicitly.

Raw JSON: `wsp/analysis/structural_analysis.json`

---

## Graph / exact fusion

| Check | Result |
|-------|--------|
| `BatchNormalization` nodes | **0** |
| Residual `Add` | present (MobileNet-style); not exact-eliminable without changing the map |
| Nonlinearities | `Relu` / `Clip` after convs — not linear-fusable |

**Hypothesis H_S1 (exact BN fusion):** **Rejected a priori.** Export already folded BN into Conv weights/biases. No exact fusion candidate that removes ops while preserving the mathematical function beyond what the exporter did.

---

## Late-backbone PW288 (five `288×288` pointwise layers)

### Effective rank (energy of singular spectrum)

| Layer | Rank@90% | Rank@95% | Rank@99% | Stable rank |
|-------|--------:|--------:|--------:|------------:|
| 626 | — | 145 | 201 | (see JSON) |
| 634 | — | 146 | 201 | |
| 642 | — | 144 | 199 | |
| 650 | — | 143 | 198 | |
| 658 | — | 123 | 181 | |
| **Mean @95%** | | **~140** | | |

Prior SVD factor **r=96** (variant C) is **below** 95% energy — it is a deliberate approximation, not an exact low-rank identity. That matches the measured −0.66 Hard pp.

Param break-even for factorization: need `r < 144`. So r=96 saves ~33% per layer; r=140 saves only ~3%.

### Filter correlation (output-filter cosine)

| Layer | Max corr | Pairs corr>0.99 |
|-------|--------:|----------------:|
| 626–658 | 0.48–0.82 | **0** |

**Hypothesis H_S2 (merge near-duplicate filters):** **No build.** Zero pairs above 0.99; no exact merge budget.

### Near-identity

All five PW288 layers: `rel_err(W, αI) ≈ 1.0`, off-diagonal energy ≈ 0.996.

**Hypothesis H_S4 (replace PW with Identity):** **Rejected** (also prior H43 Hard ≈ −23).

### Linear dependence (SV < 10⁻³ · σ_max)

At most **1–2** near-null singular values per PW288 matrix → suggested keep ≈ 286–287. Not a path to ≥20% param cut by itself.

---

## Other layers

| Tensor | Shape | Rank@90% | Rank@95% | Notes |
|--------|------:|--------:|--------:|-------|
| PW152 `610` | 152×152 | **63** | 79 | Factorization reduces params only if **r < 76** |
| Expand `618` | 288×152 | **70** | 90 | Factorization reduces params if **r ≲ 99** |
| DW288 `622…654` | 288×9 | — | — | Full-ish; last/first SV ≈ 0.3; not near-rank-1 |

---

## Channel L1 (288-stage)

p05 / median L1 score ≈ **0.92** — no large population of near-zero channels. Prior **H39** width prune to 256/224/192 collapsed Hard AP. Weak-L1 alone does not justify aggressive channel elimination.

---

## Hypotheses carried to Phase 3

| ID | Idea | Decision |
|----|------|----------|
| H_S1 | Exact BN fusion | Negative — not built |
| H_S2 | Corr>0.99 filter merge | Negative — not built |
| H_S3 | PW288 low-rank (r=96 prior) | Already validated as **C**; restacked in **S3** |
| H_S4 | PW → Identity | Negative — not rebuilt |
| H_S5 | Minimal channel elim 288→286 | Built as **S4_ch286** |
| H_S6 | PW152 r=64 (≈@90% energy) | Built as **S1_152_svd64** |
| H_S6b | Expand r=70 (≈@90% energy) | Built as **S2_expand_svd70** |
| Stack | S3 = PW288r96 + PW152r64 + expand r70 | Built as **S3_multi_svd** |

Ranks were **pre-registered from spectra / prior work**, not tuned against AP.
