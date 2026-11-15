#!/usr/bin/env python3
"""Phase-2 structural weight analysis of SCRFD-500M (weight-algebra; no training).

Produces wsp/analysis/structural_analysis.json for STRUCTURAL_ANALYSIS.md.
Does not execute the model on images (weight-algebra profile).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "det_500m.onnx"
OUT = ROOT / "wsp" / "analysis"

PW288 = ["626", "634", "642", "650", "658"]
PW152 = ["610"]
EXPAND288 = "618"
DW288 = ["622", "630", "638", "646", "654"]


def fro_rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))


def effective_rank_spectrum(W2d: np.ndarray) -> dict:
    """W2d: (Cout, Cin) for 1x1, or flattened filters."""
    s = np.linalg.svd(W2d, compute_uv=False)
    s2 = s**2
    tot = s2.sum() + 1e-30
    energy = np.cumsum(s2) / tot
    # stable rank = ||W||F^2 / ||W||2^2
    stable = float(s2.sum() / (s2[0] + 1e-30))
    # participation ratio
    p = s2 / tot
    part = float(1.0 / np.sum(p**2))
    thresh = {}
    for t in (0.90, 0.95, 0.99, 0.999):
        thresh[str(t)] = int(np.searchsorted(energy, t) + 1)
    return {
        "shape": list(W2d.shape),
        "singular_values_top16": [float(x) for x in s[:16]],
        "singular_min": float(s[-1]),
        "condition_number": float(s[0] / (s[-1] + 1e-30)),
        "stable_rank": stable,
        "participation_ratio": part,
        "rank_for_energy": thresh,
        "nuclear_norm": float(s.sum()),
        "frobenius": float(np.linalg.norm(W2d)),
    }


def filter_correlation(W: np.ndarray) -> dict:
    """Correlate output filters: W (Cout, Cin, kH, kW)."""
    Cout = W.shape[0]
    X = W.reshape(Cout, -1).astype(np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    C = Xn @ Xn.T
    np.fill_diagonal(C, 0.0)
    iu = np.triu_indices(Cout, 1)
    vals = C[iu]
    # near-duplicates
    pairs = []
    for i, j in zip(*np.where(C > 0.99)):
        if i < j:
            pairs.append((int(i), int(j), float(C[i, j])))
    pairs.sort(key=lambda t: -t[2])
    return {
        "cout": Cout,
        "corr_mean_offdiag": float(vals.mean()) if vals.size else 0.0,
        "corr_max": float(vals.max()) if vals.size else 0.0,
        "corr_p95": float(np.percentile(vals, 95)) if vals.size else 0.0,
        "corr_p99": float(np.percentile(vals, 99)) if vals.size else 0.0,
        "n_pairs_corr_gt_0.99": len(pairs),
        "n_pairs_corr_gt_0.95": int(np.sum(vals > 0.95)),
        "n_pairs_corr_gt_0.90": int(np.sum(vals > 0.90)),
        "top_pairs": pairs[:20],
    }


def linear_dependence_via_svd(W: np.ndarray, tol_ratio: float = 1e-3) -> dict:
    """Count near-zero singular values relative to largest."""
    Cout = W.shape[0]
    X = W.reshape(Cout, -1).astype(np.float64)
    s = np.linalg.svd(X, compute_uv=False)
    thr = tol_ratio * s[0]
    n_small = int(np.sum(s < thr))
    return {
        "cout": Cout,
        "tol_ratio": tol_ratio,
        "n_sv_below_tol": n_small,
        "suggested_keep": int(Cout - n_small),
        "smallest_sv_ratio": float(s[-1] / (s[0] + 1e-30)),
    }


def near_identity_pw(W: np.ndarray) -> dict:
    """Is PW close to scaled identity?"""
    C = W.shape[0]
    assert W.shape[1] == C and W.shape[2:] == (1, 1)
    M = W.reshape(C, C).astype(np.float64)
    # best scalar alpha for alpha*I
    alpha = float(np.trace(M) / C)
    rel = fro_rel(M, alpha * np.eye(C))
    # off-diagonal energy fraction
    off = M - np.diag(np.diag(M))
    off_frac = float(np.linalg.norm(off) ** 2 / (np.linalg.norm(M) ** 2 + 1e-30))
    return {"alpha_best": alpha, "rel_err_to_alpha_I": rel, "offdiag_energy_frac": off_frac}


def scan_graph_fusion(model: onnx.ModelProto) -> dict:
    """Look for exact-fusion opportunities (pattern only; no execution)."""
    ops = [n.op_type for n in model.graph.node]
    from collections import Counter

    counts = Counter(ops)
    # Conv -> Relu chains, Conv with bias already fused?, Add residuals
    patterns = {
        "Conv": 0,
        "Relu": 0,
        "Clip": 0,
        "Add": 0,
        "Resize": 0,
        "Concat": 0,
        "Mul": 0,
        "Sigmoid": 0,
        "BatchNormalization": 0,
    }
    for k in list(patterns):
        patterns[k] = counts.get(k, 0)

    # Residual Add: inputs both from Conv/Relu
    residual_adds = 0
    for n in model.graph.node:
        if n.op_type == "Add" and len(n.input) == 2:
            residual_adds += 1

    # Orphan BN?
    return {
        "op_counts": dict(counts),
        "pattern_summary": patterns,
        "n_add_nodes": residual_adds,
        "batchnorm_nodes": patterns["BatchNormalization"],
        "fusion_note": (
            "No BatchNormalization nodes in exported ONNX — BN already folded into Conv "
            "weights/biases at export. Residual Adds cannot be eliminated without changing "
            "the function. Relu/Clip after Conv are nonlinear — not exact-linear fusable."
        ),
    }


def channel_l1_redundancy(init: dict) -> dict:
    """L1 magnitudes across 288-stage channels (for pruning hypothesis, not selection search)."""
    score = np.abs(init[EXPAND288]).sum(axis=(1, 2, 3)).astype(np.float64)
    for tid in PW288:
        W = init[tid]
        score += np.abs(W).sum(axis=(1, 2, 3))
        score += np.abs(W).sum(axis=(0, 2, 3))
    for tid in DW288:
        W = init[tid]
        score += np.abs(W).reshape(W.shape[0], -1).sum(1)
    order = np.argsort(score)
    return {
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_median": float(np.median(score)),
        "score_p05": float(np.percentile(score, 5)),
        "score_p10": float(np.percentile(score, 10)),
        "weakest_10_indices": [int(i) for i in order[:10]],
        "weakest_10_scores": [float(score[i]) for i in order[:10]],
        "ratio_p05_to_median": float(np.percentile(score, 5) / (np.median(score) + 1e-30)),
        "hypothesis": (
            "If many channels have near-zero L1 across expand/DW/PW, they may be "
            "eliminable with analytic slice+compensate. Prior H39 showed L1 top-k "
            "width prune to 256/224/192 collapsed Hard AP — weak L1 alone is insufficient."
        ),
    }


def main():
    model = onnx.load(str(BASE))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    report = {
        "protocol": "structural-analysis-v1",
        "observation_profile": "weight-algebra",
        "extension_class": (
            "Structural compression (channel prune / block rewrite) is an "
            "**extension** to strict fixed-architecture WSP; declared separately."
        ),
        "baseline": "models/det_500m.onnx",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "graph_fusion_scan": scan_graph_fusion(model),
        "pw288_effective_rank": {},
        "pw288_filter_correlation": {},
        "pw288_near_identity": {},
        "pw288_linear_dependence": {},
        "pw152_effective_rank": {},
        "expand288_filter_correlation": {},
        "channel_l1_288stage": channel_l1_redundancy(init),
        "hypotheses": [],
    }

    for name in PW288:
        W = init[name]
        M = W.reshape(W.shape[0], W.shape[1])
        report["pw288_effective_rank"][name] = effective_rank_spectrum(M)
        report["pw288_filter_correlation"][name] = filter_correlation(W)
        report["pw288_near_identity"][name] = near_identity_pw(W)
        report["pw288_linear_dependence"][name] = linear_dependence_via_svd(W, 1e-3)

    W152 = init[PW152[0]]
    report["pw152_effective_rank"][PW152[0]] = effective_rank_spectrum(
        W152.reshape(W152.shape[0], W152.shape[1])
    )
    report["expand288_filter_correlation"] = filter_correlation(init[EXPAND288])

    # Summarize mean energy ranks
    r95 = [report["pw288_effective_rank"][n]["rank_for_energy"]["0.95"] for n in PW288]
    r99 = [report["pw288_effective_rank"][n]["rank_for_energy"]["0.99"] for n in PW288]
    max_corr = [
        report["pw288_filter_correlation"][n]["corr_max"] for n in PW288
    ]
    n_dup = [
        report["pw288_filter_correlation"][n]["n_pairs_corr_gt_0.99"] for n in PW288
    ]

    report["hypotheses"] = [
        {
            "id": "H_S1_exact_bn_fusion",
            "claim": "Exported ONNX still has unfused BN that can be folded exactly.",
            "prediction": "If BatchNormalization nodes exist, fold into Conv without AP change.",
            "weight_evidence": report["graph_fusion_scan"],
            "status": "likely_negative_a_priori"
            if report["graph_fusion_scan"]["batchnorm_nodes"] == 0
            else "candidate",
        },
        {
            "id": "H_S2_near_dup_filter_merge",
            "claim": "Highly correlated PW288 output filters (corr>0.99) can be merged with exact linear compensation of downstream weights.",
            "prediction": f"max_corr={max(max_corr):.4f}, n_pairs>0.99 sum={sum(n_dup)}; if near zero, no merge budget.",
            "weight_evidence": {"max_corr_per_layer": max_corr, "n_dup_pairs": n_dup},
            "status": "test_if_pairs_exist",
        },
        {
            "id": "H_S3_effective_rank_channel_slice",
            "claim": "PW288 matrices have effective rank << 288; dropping trailing singular directions as channel factors (already validated as SVD factor H41b r=96) OR slicing low-energy output channels with compensation may cut params.",
            "prediction": f"mean rank@95% energy={np.mean(r95):.1f}, @99%={np.mean(r99):.1f}; r=96 is below 99% energy — approximation not exact.",
            "weight_evidence": {"rank95": r95, "rank99": r99},
            "status": "svd_factor_already_validated_as_C; channel_slice_is_different_and_risky",
        },
        {
            "id": "H_S4_pw_near_identity_block_skip",
            "claim": "Some PW288 blocks ≈ αI and can be replaced by scaled Identity (exact if residual path absorbs).",
            "prediction": "If rel_err_to_alpha_I << 1e-2, try Identity; prior H43 Identity replace failed Hard~-23.",
            "weight_evidence": {n: report["pw288_near_identity"][n] for n in PW288},
            "status": "likely_negative",
        },
        {
            "id": "H_S5_weak_l1_channel_elim",
            "claim": "Lowest-L1 288-stage channels can be removed with analytic reshape of adjacent tensors.",
            "prediction": "Prior H39 failed for aggressive keep; only try tiny cut (e.g. 288→280) if p05/median is very small.",
            "weight_evidence": report["channel_l1_288stage"],
            "status": "cautious_candidate_small_cut_only",
        },
        {
            "id": "H_S6_pw152_svd_factor",
            "claim": "152×152 PW also admits low-rank factorization with modest error.",
            "prediction": "If rank@95% << 152, factor at that rank; expect smaller absolute savings than 288 stage.",
            "weight_evidence": report["pw152_effective_rank"],
            "status": "candidate",
        },
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "structural_analysis.json"
    path.write_text(json.dumps(report, indent=2))
    print("Wrote", path)
    print("PW288 rank@95%:", r95, "mean", float(np.mean(r95)))
    print("PW288 rank@99%:", r99)
    print("max filter corr:", max_corr, "dup pairs:", n_dup)
    print("BN nodes:", report["graph_fusion_scan"]["batchnorm_nodes"])
    for n in PW288:
        ni = report["pw288_near_identity"][n]
        print(f"  {n} nearI rel={ni['rel_err_to_alpha_I']:.3f} off={ni['offdiag_energy_frac']:.3f}")
    print("152 rank@95:", report["pw152_effective_rank"]["610"]["rank_for_energy"]["0.95"])
    print("L1 p05/med:", report["channel_l1_288stage"]["ratio_p05_to_median"])


if __name__ == "__main__":
    main()
