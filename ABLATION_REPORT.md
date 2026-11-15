# Ablation: SVD compression vs KPS-head removal

**Protocol:** `ablation-svd-vs-kps-v1`  
**Baseline checkpoint:** `models/det_500m.onnx`  
**sha256:** `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`  
**Hardware (this run):** AMD Ryzen 7 5800H, 16 logical CPUs, Linux, onnxruntime 1.24.4, `CPUExecutionProvider`, input `1×3×640×640`  
**Measured:** 2025–2026 session; full JSON in `wsp/analysis/ablation_benchmark.json`

This report isolates how much of the previously advertised compression comes from **weight-space SVD (rank 96)** versus **removing the keypoint (KPS) head**. All four variants were rebuilt from the same immutable baseline with **no training**.

---

## 1. Variants

| ID | Name | Construction | KPS outputs | Artifact |
|----|------|--------------|-------------|----------|
| **A** | Baseline | Original SCRFD-500M KPS | 9 outs (yes) | `models/ablation/A_baseline.onnx` |
| **B** | KPS-only | Strip KPS head/outputs only | 6 outs (no) | `models/ablation/B_kps_only.onnx` |
| **C** | SVD-only | Rank-96 SVD on five late `288×288` PW layers | 9 outs (yes) | `models/ablation/C_svd_only.onnx` |
| **D** | Combined | SVD rank-96 + KPS strip | 6 outs (no) | `models/ablation/D_svd_nokps.onnx` |

Integrity checks (executed):

- `A` byte-identical to `models/det_500m.onnx`.
- `D` byte-identical to previously shipped `models/compressed/det_500m_svd96_nokps.onnx`.
- `C` retains 6 KPS initializers and 9 graph outputs.
- `B`/`D` have 0 KPS initializers and 6 graph outputs.

---

## 2. Four-way results (measured)

### Parameters and size

| Variant | Total params | Backbone numeric | SVD factors | Neck | Det head (cls/reg) | KPS head | File bytes |
|---------|-------------:|-----------------:|------------:|-----:|-------------------:|---------:|-----------:|
| A | 626,332 | 549,954 | 0 | 24,448 | 17,310 | 34,620 | 2,524,817 |
| B | 591,710 | 549,952 | 0 | 24,448 | 17,310 | 0 | 2,385,065 |
| C | 488,092 | 135,234 | 276,480 | 24,448 | 17,310 | 34,620 | 1,972,636 |
| D | 453,470 | 135,232 | 276,480 | 24,448 | 17,310 | 0 | 1,832,884 |

### Compute (Conv GMACs @ 640²)

Counting: `N·Cout·Cin·kH·kW·Hout·Wout` per Conv (same convention for all).

| Variant | GMACs | #Conv nodes |
|---------|------:|------------:|
| A | 0.734 | 60 |
| B | 0.637 | 57 |
| C | 0.678 | 65 |
| D | 0.581 | 62 |

KPS removal reduces MACs because the three stride KPS `3×3` heads are real Conv compute—not postprocess-only. SVD reduces MACs less than params (each PW becomes two thinner convs).

### Latency

**ORT `session.run` only** (neural forward; warm-up 20, timed 100):

| Variant | median ms | p95 ms |
|---------|----------:|-------:|
| A | 6.26 | 9.88 |
| B | 5.13 | 8.28 |
| C | 6.55 | 15.61 |
| D | 9.40 | 15.35 |

**`SCRFD.detect` end-to-end** (preprocess + forward + decode + NMS [+ KPS decode if present]; warm-up 5, timed 30):

| Variant | Pipeline | median ms | p95 ms |
|---------|----------|----------:|-------:|
| A | detect + landmarks | 51.63 | 61.01 |
| B | detect only | 43.82 | 52.84 |
| C | detect + landmarks | 13.20 | 26.24 |
| D | detect only | 12.00 | 14.75 |

**Do not treat A/C e2e times as equivalent to B/D.** A and C still decode landmarks; B and D do not. Also, e2e gaps are **not** explained by ORT forward alone (SVD does not speed up `session.run` on this CPU in this run).

### WIDER FACE AP (DEV split, 470 images)

Same harness thresholds: score 0.02, NMS 0.45, IoU 0.5, letterbox 640.

| Variant | Easy | Medium | Hard | Δ Hard vs A |
|---------|-----:|-------:|-----:|------------:|
| A | 91.20 | 87.27 | 67.06 | — |
| B | 91.20 | 87.27 | 67.06 | **0.00** |
| C | 90.35 | 86.80 | 66.39 | **−0.66** |
| D | 90.35 | 86.80 | 66.39 | **−0.66** |

KPS removal does not change box AP (expected). All Hard drop from the combined model is attributable to **SVD**, not KPS strip.

### Landmark accuracy

**Not measured.** WIDER Face official mats in this workspace provide box difficulty flags only—no landmark GT. Variants A/C expose KPS outputs; B/D do not. Capability ≠ accuracy.

### Historical CONFIRMATION (not re-run for B/C)

| Variant | Easy | Medium | Hard | Status |
|---------|-----:|-------:|-----:|--------|
| A | 90.65 | 88.37 | 68.59 | historical `r041` |
| D | 90.22 | 88.01 | 67.88 | historical `r080` |
| B, C | — | — | — | **not executed** |

---

## 3. Isolated contributions

Absolute = after − before. Percent = relative to the “before” side. AP in percentage points (pp).

| Transition | Params | Params % | Bytes % | GMACs % | ORT median % | Detect e2e % | Hard AP (pp) | Pipelines comparable? |
|------------|-------:|---------:|--------:|--------:|-------------:|-------------:|-------------:|:---------------------:|
| **A→B** KPS alone | −34,622 | −5.5% | −5.5% | −13.2% | −17.9% | −15.1% | 0.00 | no (landmarks removed) |
| **A→C** SVD alone | −138,240 | −22.1% | −21.9% | −7.5% | +4.7% | −74.4%* | −0.66 | yes (both have KPS) |
| **A→D** Combined | −172,862 | −27.6% | −27.4% | −20.7% | +50.2% | −76.8%* | −0.66 | no |
| **B→D** SVD after KPS | −138,240 | −23.4% | −23.2% | −8.7% | +83.0% | −72.6%* | −0.66 | yes (both detect-only) |
| **C→D** KPS after SVD | −34,622 | −7.1% | −7.1% | −14.3% | +43.4% | −9.1% | 0.00 | no |

\*Large e2e speedups for SVD variants are **observed** but **not explained by ORT forward GMACs/latency** on this machine. Treat e2e as a composite metric (NMS load / score distribution / decode), not as validated neural-forward speedup.

**Follow-up (appended, does not rewrite tables above):** a controlled investigation on real WIDER images with fixed thread counts showed the prior multi× e2e gap was dominated by **post-threshold candidate count → NMS cost**, especially under the original **synthetic-noise** timing inputs. See **[LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md)**.

Additive check (params):  
`(A→B) + (A→C) = −34,622 − 138,240 = −172,862 = (A→D)` ✓

---

## 4. Verified conclusions

1. **Parameter compression is mostly SVD.** Of the −27.6% params in D vs A, **−22.1 pp** are from SVD and **−5.5 pp** from KPS removal (exactly additive).

2. **Detection AP change is entirely from SVD.** KPS strip (A→B, C→D) changes Hard AP by **0.00 pp**. SVD (A→C, B→D) changes Hard by **−0.66 pp** on DEV.

3. **KPS removal does reduce backbone-adjacent Conv compute** (head MACs), not only Python postprocess: A→B GMACs **−13.2%**.

4. **SVD reduces params more than MACs** (−22.1% params vs −7.5% GMACs for A→C) because each dense PW is replaced by two thinner convs.

5. **Neural forward latency gain from SVD is not validated on this CPU.** In this controlled run, ORT median for C and D is **not faster** than A. A prior blog-style claim of ~1.35× ORT speedup for the combined model is **superseded** by this ablation and must not be treated as confirmed.

6. **Functionality:** A/C remain landmark-capable; B/D are detection-only. Comparing their e2e latency as equal tasks is incorrect.

---

## 5. Machine-readable outputs

- `BENCHMARK_RESULTS.csv` — per-variant table  
- `BENCHMARK_CONTRIBUTIONS.csv` — A→B / A→C / A→D / B→D / C→D  
- `wsp/analysis/ablation_benchmark.json` — full raw measurements  
- `wsp/analysis/ablation_manifest.json` — variant hashes  
- `EXPERIMENT_AUDIT.md` — protocol / limitations  
