# SCRFD-500M parameter compression (no training)

Weight-space **COMPRESS** history after Hard-AP repair failed to transfer.
Structural / channel-prune work is a **declared extension** to strict
fixed-architecture WSP.

For the current validated summary see **[README.md](README.md)**.  
Ablation: [ABLATION_REPORT.md](ABLATION_REPORT.md) · Latency:
[LATENCY_INVESTIGATION.md](LATENCY_INVESTIGATION.md) ·
[PHASE1_LATENCY_BENCHMARK.csv](PHASE1_LATENCY_BENCHMARK.csv).

---

## Historical selected artifact (variant D = SVD r=96 + KPS strip)

| | Baseline (A) | **H44b / D** |
|--|--:|--:|
| **Params** | 626,332 | **453,470 (−27.6%)** |
| **Conv GMACs** @640 | 0.734 | **0.581 (−20.7%)** |
| **Landmarks** | yes | **no (detection-only)** |
| **CONFIRMATION E/M/H** | 90.65 / 88.37 / 68.59 | 90.22 / 88.01 / 67.88 |

Artifact: [`models/compressed/det_500m_svd96_nokps.onnx`](models/compressed/det_500m_svd96_nokps.onnx)

### Accuracy (historical measurements; preserved)

| Split | Easy | Medium | Hard | Δ Hard vs base |
|-------|-----:|-------:|-----:|---------------:|
| DEV (470) | 90.35 | 86.80 | 66.39 | −0.67 |
| CONFIRMATION (2756) | 90.22 | 88.01 | 67.88 | −0.71 |
| Full WIDER val (3226) | 90.24 | 87.84 | 67.67 | ~−0.7 |

Baseline DEV: 91.20 / 87.27 / 67.06 · CONFIRMATION: 90.65 / 88.37 / 68.59

---

## Isolated contributions (ablation — verified)

| Transition | Params | GMACs | DEV Hard | Functionality |
|------------|-------:|------:|---------:|---------------|
| **A→B** KPS alone | −5.5% | −13.2% | 0.00 pp | loses landmarks |
| **A→C** SVD alone | −22.1% | −7.5% | −0.66 pp | keeps landmarks |
| **A→D** Combined | −27.6% | −20.7% | −0.66 pp | loses landmarks |

Param deltas are additive. Hard AP change is **entirely** from SVD.

---

## Latency — finalized (Phase-1)

**Protocol:** 96 real WIDER DEV images, `intra_op_num_threads=4`,
`inter_op_num_threads=1`, ORT 1.24.4, CPU Ryzen 7 5800H. Bootstrap 95% CI on
ORT median.

| Variant | ORT median [CI] | e2e detect median | Pipeline |
|---------|----------------:|------------------:|----------|
| A | **8.12** [8.08, 8.16] | 21.99 | detect+landmarks |
| B | **6.27** [6.26, 6.29] | 19.43 | detect_only |
| C | **8.15** [8.04, 8.29] | 21.48 | detect+landmarks |
| D | **5.98** [5.96, 6.01] | 18.74 | detect_only |

**SVD-only (C) does not beat A on ORT forward.** D’s ORT win vs A mixes KPS
removal with SVD; apples-to-apples landmark-preserving compare is A vs C (~0%).

### Superseded historical latency claims

| Claim | Status |
|-------|--------|
| ORT ~1.35× for H44b vs baseline | **Superseded** — not replicated under fixed threads |
| Ablation e2e A~52 ms vs C/D~12 ms on random noise | **Artifact** — noise post-thresh counts (~1313 vs ~84); see latency investigation |

---

## Method (combined artifact)

1. **SVD low-rank PW factorization (H41b / C)** — five late `288×288` PW convs →
   `B(r×288)→A(288×r)`, **r=96**, no fine-tuning. KPS retained.
2. **Strip KPS head (H44 / B)** — removes landmark outs (~35k params, −13% GMACs).
3. **Combined (H44b / D)** = (1)+(2). Detection-only product.

```bash
python wsp/build_ablation_variants.py
python wsp/run_phase1_latency_benchmark.py
```

---

## What failed (honest negatives)

| Attempt | Idea | Outcome |
|---------|------|---------|
| H39 | Structured width prune 288→256/224/192 | Hard −4.5 to −18 |
| H40/H42 | Delete entire 288-ch DW+PW blocks | Hard ~−22 |
| H43 | Replace PW with Identity | Hard ~−23 |
| H41c | SVD rank 64 on PW288 | Easy/Hard outside ±2 DEV budget |

Phase-2 weight analysis (no near-duplicate filters, no unfused BN, PW not ≈ I)
is in [STRUCTURAL_ANALYSIS.md](STRUCTURAL_ANALYSIS.md).

---

## Notes

- Protocol: no gradients / training / distillation / output regression.
- Landmark **accuracy** unmeasured (no GT in workspace).
- B/C CONFIRMATION AP not re-run in ablation — do not invent.
