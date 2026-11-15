# Compression experiments — Phase 3 structural candidates

**Protocol:** `structural-candidates-v1`  
**Profile:** Structural compression as a **declared extension** to strict fixed-architecture WSP  
**Constraints:** no training, gradients, distillation, output regression, or AP-driven rank search  
**KPS:** retained on all new candidates (fair vs baseline)

Manifest: `models/structural/manifest.json`  
Builder: `wsp/build_structural_candidates.py`

---

## Pre-registered candidates

| ID | Hypothesis | Transformation | Params | Δ vs A | KPS |
|----|------------|----------------|-------:|-------:|:---:|
| **S1_152_svd64** | H_S6 | SVD-factor PW `610` (152×152) at **r=64** (≈rank@90% energy) | 622,684 | −0.58% | yes |
| **S2_expand_svd70** | H_S6b | SVD-factor expand `618` (288×152) at **r=70** | 613,356 | −2.07% | yes |
| **S3_multi_svd** | H_S3+H_S6 stack | PW288 **r=96** + PW152 r=64 + expand r=70 | 471,468 | **−24.7%** | yes |
| **S4_ch286** | H_S5 | Analytic channel slice 288→286 (weakest L1) | 620,144 | −0.99% | yes |

### Mathematical form (SVD factor)

For a pointwise weight \(W\in\mathbb{R}^{C_\mathrm{out}\times C_\mathrm{in}}\):

\[
W \approx A B,\quad
B=\mathrm{diag}(\sqrt{\sigma_{1:r}})\,V_{1:r}^\top,\quad
A=U_{1:r}\,\mathrm{diag}(\sqrt{\sigma_{1:r}})
\]

Replaces one `Conv` with two thinner `Conv`s. Truncation is weight-algebra only (no fitting to activations).

### Channel slice (S4)

Keep index set \(K\) = top-286 L1 scores over expand/DW/PW/lateral-in. Slice:

- expand out, DW out, PW in+out, lateral in  
- set DW `group=|K|`

---

## Reconstruction

```bash
python wsp/structural_analysis.py          # Phase 2 (weights only)
python wsp/build_structural_candidates.py  # writes models/structural/S*.onnx
python wsp/run_structural_benchmark.py     # Phase 4
```

Baseline immutable: `models/det_500m.onnx`.

---

## Negatives (not built as new ONNX)

| Idea | Evidence | Outcome |
|------|----------|---------|
| Exact BN fusion | 0 BN nodes in ONNX | Nothing to fold |
| Merge filters corr>0.99 | 0 pairs in PW288 | No merge budget |
| PW → Identity | rel_err(αI)≈1; prior H43 | Not rebuilt |
| Aggressive width prune | Prior H39 Hard −4.5…−18 | Not repeated |

---

## Reference models (not new)

| ID | Role |
|----|------|
| A_baseline | Original SCRFD-500M KPS |
| C_svd_only | Existing PW288 r=96, KPS kept |
| D_svd_nokps | Existing SVD+KPS strip — **detection-only** (labeled; not a fair KPS-preserving compare) |

---

## DEV outcomes (summary)

Full tables: [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md).

| Candidate | Meets ≥20% params | Meets ≥15% ORT↓ | Hard drop ≤1 pp | All gates |
|-----------|:-----------------:|:---------------:|:---------------:|:---------:|
| S1 | no | no (slightly slower) | **no** (−3.08) | **no** |
| S2 | no | no (slower) | yes (−0.27) | **no** |
| S3 | **yes** | no (slower) | **no** (−4.72) | **no** |
| S4 | no | no (slower) | yes (−0.11) | **no** |

**No candidate frozen for CONFIRMATION.** Reporting **negative / inconclusive** vs research targets rather than promoting a weaker model.
