# Latency investigation — why SVD looked much faster end-to-end

**Protocol:** `latency-investigation-v1`  
**Date:** 2026-11-15  
**Hardware:** AMD Ryzen 7 5800H (16 logical CPUs), Linux, onnxruntime **1.24.4**, `CPUExecutionProvider`  
**Inputs:** 24 real WIDER FACE DEV images (same set for A/B/C/D) + controlled synthetic-noise contrast  
**Artifacts:** `wsp/analysis/latency_investigation.json`, `latency_ort_threads.csv`, `latency_noise_vs_real.json`  
**Constraint:** no weight edits, no training; prior ablation results preserved

This note explains the ablation’s large **detect() e2e** gap (A/B ≫ C/D) while **ORT forward** did not show a matching win.

---

## Short answer

**SVD does not deliver a large, validated neural-forward speedup on this CPU.**

The dramatic prior e2e advantage for SVD variants was driven mainly by **far fewer post-threshold candidates → much cheaper NMS**, especially on **synthetic random images** used in the original ablation timing. When thread count is fixed, order is randomized, and postprocess candidate budgets are equalized on **real WIDER images**, that large gap **disappears**. Remaining differences are mostly:

1. **KPS head present vs absent** (extra Conv outputs + ~0.45 ms KPS decode), and  
2. **Mild score-sparsity differences** that change NMS cost under the default `score≥0.02` pipeline — not a faster backbone matmul.

---

## 1. ORT forward — fixed threads, shuffled order, real images

Method: letterbox each of 24 WIDER images → shared blobs; for threads ∈ {1,2,4,8}, 3 trials with **randomized variant order**; warm-up then 3 timed `session.run` calls per image. `intra_op_num_threads` fixed; `inter_op_num_threads=1`.

| Threads | A baseline | B KPS-only | C SVD-only | D SVD+noKPS |
|--------:|-----------:|-----------:|-----------:|------------:|
| 1 | 23.89 | 20.22 | 22.67 | 18.98 |
| 2 | 13.82 | 11.39 | 13.13 | 10.67 |
| 4 | 8.38 | 6.60 | 7.91 | 6.14 |
| 8 | 6.03 | 4.14 | 5.49 | 4.40 |

Values: median ms over all timed runs (trial medians were stable within ~0.3–0.5 ms).

**Relative ORT (vs A) at 4 threads:** B **−21%**, C **−6%**, D **−27%**.

Interpretation:

- Removing KPS outputs (A→B, C→D) consistently speeds ORT more than SVD alone.
- SVD alone (A→C) is a **small** ORT improvement here (~5–9% across thread settings), not ~1.3–4×.
- Prior ablation’s ORT medians (A 6.26 / D 9.40 under default threads, synthetic blob, fixed order) were **run-order / thread-default sensitive**; this controlled sweep does **not** reproduce “D slower than A” as a stable fact, nor a large SVD-only win.

---

## 2. Stage profile on real WIDER images (threads=4)

Stages timed separately (median ms over 24 images × 3 repeats):

| Stage | A | B | C | D |
|-------|--:|--:|--:|--:|
| Letterbox preprocess (shared) | ~1.1 | ~1.1 | ~1.1 | ~1.1 |
| `blobFromImage` | 2.33 | 2.33 | 2.36 | 2.38 |
| ORT `session.run` | 9.78 | 7.97 | 9.50 | 7.66 |
| Box decode (all anchors) | 0.16 | 0.16 | 0.17 | 0.16 |
| KPS decode (all anchors) | 0.45 | 0.00 | 0.45 | 0.00 |
| NMS | 6.80 | 6.80 | 5.11 | 5.28 |
| Total post (decode+NMS+…) | 7.55 | 7.14 | 5.85 | 5.58 |
| Blob+ORT+post | 19.97 | 17.62 | 17.96 | 16.67 |

Full `SCRFD.detect` e2e on the same 24 images (threads=4): A **19.8** / B **18.0** / C **20.0** / D **16.4** ms median — **no multi× SVD win**.

KPS decode is real but small (~0.45 ms). **NMS is a first-class cost** on this pipeline at `thresh=0.02`.

---

## 3. Detection counts (before / after threshold & NMS)

Same 24 real images, `score ≥ 0.02`, NMS 0.45:

| Variant | Anchors | Post-thresh (median) | Post-thresh (mean) | Post-NMS (median) |
|---------|--------:|---------------------:|-------------------:|------------------:|
| A | 16800 | 728 | 1146 | 302 |
| B | 16800 | 728 | 1146 | 302 |
| C | 16800 | 587 | 1120 | 232 |
| D | 16800 | 587 | 1120 | 232 |

A≡B and C≡D on counts (expected: KPS strip does not change box scores).  
SVD variants produce **fewer** candidates above 0.02 (median 587 vs 728), so NMS is cheaper.

**Pearson r (n_post_thresh vs nms_ms across image×variant means) = 0.93.**  
NMS time tracks candidate count, not SVD topology.

---

## 4. Synthetic noise vs real images (explains the old ablation)

Original ablation `bench_detect` used **uniform random uint8** frames. Contrast (8 noise frames vs 16 real; threads=4):

| Input | Variant | Post-thresh median | NMS median ms | ORT+post median ms |
|-------|---------|-------------------:|--------------:|-------------------:|
| Noise | A | **1313** | **25.8** | **37.0** |
| Noise | B | **1313** | **25.7** | **34.4** |
| Noise | C | **84** | **1.3** | **11.3** |
| Noise | D | **84** | **1.3** | **8.8** |
| Real | A | 578 | 5.2 | 15.7 |
| Real | B | 578 | 5.1 | 14.0 |
| Real | C | 381 | 3.2 | 13.2 |
| Real | D | 381 | 3.3 | 11.8 |

On noise, baseline emits ~**16×** more post-thresh boxes than SVD models → NMS dominates → e2e looks like a huge “SVD speedup” while ORT stays ~8–10 ms. That matches the prior ablation pattern (A ~52 ms vs C/D ~12–13 ms e2e) and is **not** a neural-forward result.

---

## 5. Controlled postprocess — equal candidate budgets

Keep only top-K scores before NMS (same K for all variants). Median NMS and ORT+post (threads=4, 24 real images):

| Cap K | A ORT+post | B | C | D | NMS (all ≈) |
|------:|-----------:|--:|--:|--:|------------:|
| 50 | 10.60 | 8.18 | 10.31 | 7.92 | ~0.21 ms |
| 100 | 10.83 | 8.49 | 10.55 | 8.24 | ~0.45 ms |
| 200 | 11.59 | 9.21 | 11.22 | 8.95 | ~1.1 ms |
| 500 | 13.13 | 10.74 | 13.13 | 10.57 | ~2.5 ms |
| natural | 17.77 | 15.14 | 15.68 | 13.39 | 5–7 ms |

Once K is equalized, **NMS times match across A/B/C/D**. The leftover ORT+post gap is essentially **KPS vs no-KPS ORT** (A≈C, B≈D). **SVD’s natural-cap e2e edge does not survive equalized postprocess workload.**

---

## 6. Updated conclusion

| Claim | Verdict |
|-------|---------|
| SVD largely reduces params/GMACs (from ablation) | Still valid — unchanged |
| SVD causes the DEV Hard AP drop | Still valid — unchanged |
| SVD yields large e2e `detect()` speedup | **Rejected as a network-speed claim.** Artifact of candidate count / NMS (worst on random noise) |
| SVD yields large ORT `session.run` speedup | **Not validated.** At fixed threads on real images: ~5–9% vs A; KPS removal helps more |
| Prefer params + GMACs + matched-pipeline ORT for compress quality | **Reinforced** |

**Practical guidance:** report latency with fixed `intra_op_num_threads`, real images, stage breakdown, and post-thresh counts. Do not compare detect+landmarks vs detect-only e2e as equal work. Do not use synthetic noise for detector e2e timing.

---

## Files

| Path | Role |
|------|------|
| `wsp/run_latency_investigation.py` | Reproducible investigation harness |
| `wsp/analysis/latency_investigation.json` | Full measurements |
| `wsp/analysis/latency_ort_threads.csv` | ORT thread-sweep table |
| `wsp/analysis/latency_noise_vs_real.json` | Noise vs real candidate/NMS contrast |
| `ABLATION_REPORT.md` / `BENCHMARK_RESULTS.csv` | **Preserved** prior ablation numbers |
