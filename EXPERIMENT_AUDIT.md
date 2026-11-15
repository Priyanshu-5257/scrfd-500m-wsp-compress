# Experiment audit — SVD vs KPS ablation

**Protocol ID:** `ablation-svd-vs-kps-v1`  
**Reports:** [ABLATION_REPORT.md](ABLATION_REPORT.md) · [BENCHMARK_RESULTS.csv](BENCHMARK_RESULTS.csv) · [BENCHMARK_CONTRIBUTIONS.csv](BENCHMARK_CONTRIBUTIONS.csv)  
**Raw:** `wsp/analysis/ablation_benchmark.json`, `ablation_contributions.json`, `ablation_manifest.json`

---

## 1. Protocol compliance

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Four variants A/B/C/D from same baseline | **Pass** | `wsp/build_ablation_variants.py`; A sha = baseline |
| No train / fine-tune / distill / gradients | **Pass** | Construction is ONNX graph rewrite + NumPy SVD only |
| No new hyperparameter search | **Pass** | Rank fixed at previously selected r=96 |
| Same preprocessing / eval thresholds | **Pass** | letterbox 640, score 0.02, NMS 0.45, IoU 0.5 via `wsp/harness.py` |
| Checkpoint integrity | **Pass** | A ≡ `models/det_500m.onnx`; D ≡ shipped H44b ONNX |
| SVD-only preserves KPS architecture | **Pass** | C: 9 outputs, 6 KPS initializers |
| KPS removal impact on compute characterized | **Pass** | B/D drop 3 Conv nodes; GMACs −13% vs peers with KPS |
| Separate ORT forward vs e2e detect | **Pass** | Both reported; pipelines labeled |
| Historical manifests preserved | **Pass** | Prior `compress_outcome.json`, H44b logs untouched; new ablation files added |
| Do not claim unexecuted experiments | **Pass** | B/C CONFIRMATION marked `not_executed`; landmark GT unavailable |

---

## 2. Reproducibility recipe

```bash
# 1) Build variants (from immutable baseline)
python wsp/build_ablation_variants.py

# 2) Benchmark + DEV AP for A–D
python wsp/run_ablation_benchmark.py

# Artifacts written under models/ablation/ and wsp/analysis/ablation_*.json
# CSV tables: BENCHMARK_RESULTS.csv, BENCHMARK_CONTRIBUTIONS.csv
```

**Environment recorded for this run**

| Field | Value |
|-------|-------|
| CPU | AMD Ryzen 7 5800H (16 logical) |
| OS | Linux (glibc 2.43) |
| Python | 3.12.12 |
| onnxruntime | 1.24.4 |
| Provider | `CPUExecutionProvider` |
| Threads | ORT default (`intra_op_num_threads` unset) |
| Input | `1×3×640×640` |
| ORT timing | warm-up 20, timed 100 → median / p95 |
| Detect timing | warm-up 5, timed 30 → median / p95 |
| AP split | DEV 470 images (`wsp/splits/dev.json`) |

**Baseline sha256:** `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a`

| Variant | sha256 |
|---------|--------|
| A | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| B | `9e62c79fa95844496576c2849fcd042754a21ea136c38a37acc77e485c9a197c` |
| C | `f1ac6ab864d347f5257f8e03682efd6713418262f5a9804b7eb47e3f26b4fb9f` |
| D | `ebcf9b9ffaf41d338df65867ab99ce87ede7c5b87d01217e6c1e3235b3866a55` |

---

## 3. What was measured vs previously reported

| Claim | Source | Ablation status |
|-------|--------|-----------------|
| Combined −27.6% params, −20.7% GMACs | H44b / COMPRESS.md | **Confirmed** (A→D) |
| Combined Hard ≈ −0.7 on DEV/CONF | H44b | DEV **confirmed** (−0.66); CONF historical for A/D only |
| KPS strip does not change box AP | H44 narrative | **Confirmed** (A=B, C=D on DEV) |
| ORT ~1.35× faster for combined vs baseline | Prior `COMPRESS.md` / H44b latency note | **Not replicated.** Ablation ORT medians: A 6.26 ms, D 9.40 ms. Prior figure superseded. |
| SVD alone preserves landmarks | Implicit | **Confirmed** (C has KPS outs) |
| Landmark numeric accuracy | Never measured | **Unavailable** (no GT) |
| B/C CONFIRMATION AP | Not in prior runs | **Not executed** — do not invent |

---

## 4. Measurement limitations

1. **Single-machine CPU latency.** No GPU, no multi-run day-to-day variance study. ORT thread pinning was default, not locked to 1.
2. **E2E `detect()` is composite.** Includes letterbox, decode, NMS, and optional KPS. Large SVD-related e2e drops are **not** mirrored in ORT `session.run` and must not be sold as pure network speedup.
3. **KPS vs no-KPS pipelines are different products.** Latency and “speedup” across that axis is not apples-to-apples functionality.
4. **CONFIRMATION not re-run for B and C.** Isolation of AP on holdout is DEV-only for the four-way design; A/D CONF numbers are historical.
5. **Landmark accuracy untested.** Capability flags only.
6. **MAC counting is Conv-only** under a fixed formula; Gemm/other ops (if any) are not included the same way—SCRFD-500M here is Conv-dominated.
7. **Prior ORT speedup claim** used a different timing context than this controlled A–D sweep; treat ablation ORT numbers as the authoritative comparison for SVD vs KPS isolation.

---

## 5. Structural findings (correctness)

- **KPS removal** deletes three stride-level KPS Conv heads and their graph outputs. That **does** reduce Conv GMACs (~13%), not merely Python postprocess. Backbone stem/torso weights are unchanged (aside from tiny constant-name bookkeeping).
- **SVD-only** replaces five late `288→288` PW Convs with `288→96` then `96→288` pairs; KPS heads remain. Param drop (−138,240) is identical for A→C and B→D.
- Combined params equal the sum of the two isolated deltas (additive on this graph).

---

## 6. Latency follow-up (appended)

Protocol `latency-investigation-v1` (`LATENCY_INVESTIGATION.md`, `wsp/analysis/latency_investigation.json`):

- Fixed threads {1,2,4,8}, shuffled variant order, 24 real WIDER DEV images.
- Stage split: preprocess / blob / ORT / box decode / KPS decode / NMS.
- Candidate counts + top-K capped NMS equalization.
- Noise-vs-real contrast explains prior e2e artifact (baseline ~1313 post-thresh boxes on noise vs SVD ~84).

Prior ablation CSVs/JSON were **not modified**; this audit appends the latency resolution only.

## 7. Corrected final conclusion (for README / COMPRESS)

Of the shipped combined compression (A→D):

- **~80% of the parameter cut** is SVD (−138,240 / −172,862).
- **~20% of the parameter cut** is KPS removal (−34,622 / −172,862).
- **100% of the measured DEV Hard AP drop** (−0.66 pp) is from SVD; KPS removal contributes **0.00 pp**.
- **Validated neural-forward latency win from SVD alone is not established** on this CPU in the ablation run.
- Prefer GMACs + params + AP (and ORT forward with matched output arity) over e2e `detect()` when comparing compression quality.

Historical COMPRESS manifests remain as-is; this audit and `ABLATION_REPORT.md` are the corrected interpretation layer.
