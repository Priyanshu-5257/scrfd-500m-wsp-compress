# SCRFD-500M WSP pilot notes

## Contract
- Task: REPAIR
- Profile: bounded-behavior (DEV only until freeze)
- Primary: Hard AP% on DEV; preserve Easy/Medium drop ≤ 1.0
- Budget: 10 eval requests (3 reserved for confirmation)

## DEV split
- Rule: `sha256('widerface-val-v1|' + event) % 5 == 0`
- 9 events / 470 images; CONFIRMATION sealed (52 events / 2756 images)

## Baseline (r000)
- Easy/Medium/Hard DEV AP%: **91.20 / 87.27 / 67.06**

## Weight-algebra fact
- `stride_cls.(8,8).bias` ≈ [-0.450, -0.617]
- `stride_cls.(16,16).bias` ≈ [-0.117, -0.297]
- `stride_cls.(32,32).bias` ≈ [-0.212, -0.231]
- Stride-8 (small-face / hard-relevant) logits are systematically more suppressed.

## Hypothesis H1
- Mechanism: negative stride-8 cls bias under-fires small faces → Hard AP lag.
- Edit: add scalar α to both entries of `bbox_head.stride_cls.(8, 8).bias`.
- Predeclared α ∈ {0.15, 0.30, 0.45} (written before scoring).
- Predict: Hard ↑; Easy/Medium mild ↓ from extra FPs.
- Falsify: Hard flat/↓ or Easy drop > 1.0 without Hard gain ≥ 0.5.


## H1 result (falsified)
| alpha | Easy | Medium | Hard | dHard |
|---|---:|---:|---:|---:|
| 0 (base) | 91.20 | 87.27 | 67.06 | 0 |
| 0.15 | 90.90 | 87.18 | 67.07 | +0.01 |
| 0.30 | 90.43 | 86.99 | 67.02 | -0.04 |
| 0.45 | 89.92 | 86.81 | 66.97 | -0.08 |
Decision: revert. Bias shift raises FP mass without Hard gain.

## Hypothesis H2
- Mechanism: stride-8 cls weights under-amplify small-face features (out L2 already highest, but try gain).
- Edit: mul_scalar β on `bbox_head.stride_cls.(8, 8).weight`, β ∈ {1.10, 1.20} predeclared.
- Predict: Hard ↑ with less Easy damage than bias shift.
- Falsify: Hard flat/↓.


## H2 result (falsified)
| beta | Easy | Medium | Hard | dHard |
|---|---:|---:|---:|---:|
| 1.00 | 91.20 | 87.27 | 67.06 | 0 |
| 1.10 | 91.15 | 87.24 | 67.01 | -0.05 |
| 1.20 | 91.13 | 87.27 | 66.98 | -0.07 |
Decision: revert.

## H3 result (falsified for Hard; exploratory Easy/Medium lift)
Equalize stride-8 cls biases by lifting second anchor by +0.1664.
DEV AP%: 91.62 / 87.66 / 66.96 (dHard=-0.09, dEasy=+0.42, dMed=+0.39)
Decision: do **not** select for Hard objective. CONFIRMATION left sealed.

## Pilot outcome
**negative_result** — retain baseline checkpoint.
requests_used=7/10; confirmation_consumed=false.


## Failure READ (from r000 preds, no new request)
- Hard faces: hit 3694 (med size≈26.9) vs miss 914 (med size≈12.0)
- Misses concentrate in crowded tiny-face scenes (basketball/football/voting).
- Mechanism target: amplify PAFPN finest level (lateral0/fpn0/pafpn0), not cls bias.

## Contract amendment
- successful_requests: 10 → 25
- editable: + finest/mid neck FPN tensors
- CONFIRMATION still sealed

## Hypothesis H4
- Edit: mul_scalar γ on lateral0+fpn0+pafpn0 weights&biases
- Predeclared γ ∈ {1.20, 1.40, 1.60}
- Predict: Hard ↑ by recovering tiny faces; Easy/Med mild change
- Falsify: Hard flat/↓ or Easy drop >1.0


## H4 result (falsified)
Finest FPN gain γ=1.2/1.4/1.6 all hurt Hard (dH -0.9/-4.3/-10.5). Revert.

## Localization READ
Hard recall@0.5=80.2% but @0.3=94.9% → many near-misses. Prefer reg edits.

## Hypothesis H5
- Add ε to stride_reg.(8,8).bias (expands distance2bbox boxes).
- Predeclared ε ∈ {0.08, 0.16, 0.24}


## H5 result (falsified; important sign)
Expanding stride-8 boxes hurt Hard badly (dH -2.5/-9.0/-17.4). Boxes are oversized for tiny faces, not undersized.

## Hypothesis H7
- Shrink stride-8 boxes: add ε<0 to stride_reg.(8,8).bias
- Predeclared ε ∈ {-0.05, -0.10, -0.16}
