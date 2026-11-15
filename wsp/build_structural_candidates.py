#!/usr/bin/env python3
"""Phase-3 structural candidates from weight-algebra hypotheses (no training).

Extension to strict fixed-architecture WSP: graph rewrites / factorization.
All candidates **retain KPS** for fair comparison with baseline.

Candidates (pre-registered; ranks fixed from Phase-2 spectra, not searched):
  S1_152_svd64   — factor PW 610 (152×152) at r=64 ≈ rank@90% energy
  S2_expand_svd70 — factor expand 618 (288×152) at r=70 ≈ rank@90% energy
  S3_multi_svd   — PW288 r=96 + PW152 r=64 + expand r=70
  S4_ch286       — drop 2 weakest L1 channels in 288-stage (analytic slice)

Negatives documented without new builds: BN fusion (0 nodes), filter merge
(corr>0.99 pairs=0), PW→Identity (prior H43).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "models" / "det_500m.onnx"
OUT_DIR = ROOT / "models" / "structural"
ANALYSIS = ROOT / "wsp" / "analysis"

# Late PW288 factorization targets (same as svd_factor_pw.PW)
PW288 = [
    ("626", "628", "345", "346"),
    ("634", "636", "351", "352"),
    ("642", "644", "357", "358"),
    ("650", "652", "363", "364"),
    ("658", "660", "369", "370"),
]

# Mid PW152: need din/dout from graph
PW152_W, PW152_B = "610", "612"

# Expand into 288 stage
EXPAND_W, EXPAND_B = "618", "620"

# For S4 channel slice (reuse structured_prune targets)
from wsp.structured_prune import (  # noqa: E402
    DW_B,
    DW_W,
    EXPAND_B as SP_EXPAND_B,
    EXPAND_W as SP_EXPAND_W,
    LATERAL_B,
    LATERAL_W,
    PW_B,
    PW_W,
    STAGE_BIAS,
    channel_scores,
    keep_indices,
    param_count as sp_param_count,
    replace_initializer,
    set_conv_group,
)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_count(model: onnx.ModelProto) -> int:
    return int(sum(numpy_helper.to_array(t).size for t in model.graph.initializer))


def find_conv_io(model: onnx.ModelProto, weight_name: str):
    for n in model.graph.node:
        if n.op_type == "Conv" and weight_name in n.input:
            return n.input[0], n.output[0], n
    raise KeyError(weight_name)


def svd_factor_named(
    model: onnx.ModelProto,
    weight_name: str,
    bias_name: str | None,
    rank: int,
    tag: str,
) -> dict:
    """Replace one Conv weight W (Cout,Cin,1,1) with B(r,Cin) then A(Cout,r)."""
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    W = init[weight_name]
    assert W.shape[2:] == (1, 1), W.shape
    Cout, Cin = int(W.shape[0]), int(W.shape[1])
    assert rank < min(Cout, Cin) or rank <= min(Cout, Cin)
    din, dout, conv = find_conv_io(model, weight_name)
    M = W.reshape(Cout, Cin).astype(np.float64)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Wr = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    rel = float(np.linalg.norm(M - Wr) / (np.linalg.norm(M) + 1e-30))
    B = (np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]).astype(np.float32).reshape(
        rank, Cin, 1, 1
    )
    A = (U[:, :rank] * np.sqrt(S[:rank])).astype(np.float32).reshape(Cout, rank, 1, 1)
    b_name = f"{tag}_B"
    a_name = f"{tag}_A"
    mid = f"{tag}_mid"
    new_inits = [
        numpy_helper.from_array(B, name=b_name),
        numpy_helper.from_array(A, name=a_name),
    ]
    inputs2 = [mid, a_name]
    if bias_name and bias_name in conv.input:
        inputs2.append(bias_name)
    c1 = helper.make_node(
        "Conv",
        inputs=[din, b_name],
        outputs=[mid],
        name=f"{tag}_down",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=1,
    )
    c2 = helper.make_node(
        "Conv",
        inputs=inputs2,
        outputs=[dout],
        name=f"{tag}_up",
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
        group=1,
    )
    new_nodes = []
    for n in model.graph.node:
        if n is conv:
            new_nodes.extend([c1, c2])
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    keep = [t for t in model.graph.initializer if t.name != weight_name]
    keep.extend(new_inits)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    return {
        "weight": weight_name,
        "rank": rank,
        "shape": [Cout, Cin],
        "rel_frob": rel,
        "params_before_layer": int(Cout * Cin),
        "params_after_layer": int(rank * Cin + Cout * rank),
    }


def factor_pw288(model: onnx.ModelProto, rank: int) -> list[dict]:
    metas = []
    # Apply sequentially; graph changes each time — use fixed PW list via weight names
    for i, (wn, bn, din, dout) in enumerate(PW288):
        # din/dout from table still valid as tensor names
        init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        if wn not in init:
            raise RuntimeError(f"missing {wn}")
        W = init[wn].astype(np.float64).reshape(288, 288)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        Wr = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
        rel = float(np.linalg.norm(W - Wr) / np.linalg.norm(W))
        B = (np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]).astype(np.float32).reshape(
            rank, 288, 1, 1
        )
        A = (U[:, :rank] * np.sqrt(S[:rank])).astype(np.float32).reshape(288, rank, 1, 1)
        b_name = f"pw288_{i}_B_r{rank}"
        a_name = f"pw288_{i}_A_r{rank}"
        mid = f"pw288_{i}_mid_r{rank}"
        new_inits = [
            numpy_helper.from_array(B, name=b_name),
            numpy_helper.from_array(A, name=a_name),
        ]
        node_by_out = {o: n for n in model.graph.node for o in n.output}
        conv = node_by_out[dout]
        c1 = helper.make_node(
            "Conv",
            [din, b_name],
            [mid],
            name=f"pw288_{i}_down",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )
        c2 = helper.make_node(
            "Conv",
            [mid, a_name, bn],
            [dout],
            name=f"pw288_{i}_up",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )
        new_nodes = []
        for n in model.graph.node:
            if n is conv:
                new_nodes.extend([c1, c2])
            else:
                new_nodes.append(n)
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        keep = [t for t in model.graph.initializer if t.name != wn]
        keep.extend(new_inits)
        del model.graph.initializer[:]
        model.graph.initializer.extend(keep)
        metas.append({"weight": wn, "rank": rank, "rel_frob": rel})
    return metas


def build_ch286(model: onnx.ModelProto) -> dict:
    """Drop 2 weakest L1 channels (288→286) with analytic tensor slicing."""
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    score = channel_scores(init)
    keep = keep_indices(score, 286)
    dropped = [int(i) for i in range(288) if i not in set(keep.tolist())]

    def slice_out(W):
        return W[keep]

    def slice_in(W):
        return W[:, keep]

    replace_initializer(model, SP_EXPAND_W, slice_out(init[SP_EXPAND_W]))
    replace_initializer(model, SP_EXPAND_B, init[SP_EXPAND_B][keep])
    for w, b in zip(DW_W, DW_B):
        Ww = init[w][keep]
        replace_initializer(model, w, Ww)
        replace_initializer(model, b, init[b][keep])
        set_conv_group(model, w, 286)
    for w, b in zip(PW_W, PW_B):
        Ww = init[w][keep][:, keep]
        replace_initializer(model, w, Ww)
        replace_initializer(model, b, init[b][keep])
    # lateral takes 288-ch input
    replace_initializer(model, LATERAL_W, init[LATERAL_W][:, keep])
    # bias unchanged
    return {
        "keep_c": 286,
        "dropped_channels": dropped,
        "dropped_scores": [float(score[i]) for i in dropped],
    }


def save_candidate(model: onnx.ModelProto, name: str, meta: dict) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        onnx.checker.check_model(model)
        ok, err = True, None
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)
    path = OUT_DIR / f"{name}.onnx"
    onnx.save(model, str(path))
    rec = {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "params": param_count(model),
        "onnx_check_ok": ok,
        "onnx_check_err": err,
        "has_kps": True,
        "num_outputs": len(model.graph.output),
        "meta": meta,
    }
    (OUT_DIR / f"{name}.meta.json").write_text(json.dumps(rec, indent=2))
    print(
        f"Wrote {path} params={rec['params']} check={ok} sha={rec['sha256'][:12]}...",
        flush=True,
    )
    return rec


def main():
    base_params = param_count(onnx.load(str(BASE)))
    records = []

    # S1
    m = onnx.load(str(BASE))
    meta = {
        "hypothesis": "H_S6",
        "justification": (
            "PW152 (610) spectrum: rank@90% energy = 63; choose r=64 (pre-registered). "
            "r < 152/2 so factorization reduces params. Analytic SVD truncation only."
        ),
        "layers_changed": [PW152_W],
        "factor": svd_factor_named(m, PW152_W, PW152_B, 64, "pw152_r64"),
    }
    meta["params_before"] = base_params
    meta["params_after"] = param_count(m)
    records.append(save_candidate(m, "S1_152_svd64", meta))

    # S2
    m = onnx.load(str(BASE))
    meta = {
        "hypothesis": "H_S6_expand",
        "justification": (
            "Expand 618 (288×152) rank@90% energy = 70; factor at r=70. "
            "Break-even r≈99; r=70 reduces params. KPS untouched."
        ),
        "layers_changed": [EXPAND_W],
        "factor": svd_factor_named(m, EXPAND_W, EXPAND_B, 70, "expand_r70"),
    }
    meta["params_before"] = base_params
    meta["params_after"] = param_count(m)
    records.append(save_candidate(m, "S2_expand_svd70", meta))

    # S3 stack
    m = onnx.load(str(BASE))
    m288 = factor_pw288(m, 96)
    f152 = svd_factor_named(m, PW152_W, PW152_B, 64, "pw152_r64")
    fexp = svd_factor_named(m, EXPAND_W, EXPAND_B, 70, "expand_r70")
    meta = {
        "hypothesis": "H_S3+H_S6 stack",
        "justification": (
            "Independent spectrum-motivated factorizations stacked: "
            "PW288 r=96 (prior H41b), PW152 r=64, expand r=70. "
            "No coefficient search; ranks fixed from Phase-2 energy tables / prior."
        ),
        "layers_changed": [x[0] for x in PW288] + [PW152_W, EXPAND_W],
        "pw288": m288,
        "pw152": f152,
        "expand": fexp,
    }
    meta["params_before"] = base_params
    meta["params_after"] = param_count(m)
    records.append(save_candidate(m, "S3_multi_svd", meta))

    # S4 channel elim
    m = onnx.load(str(BASE))
    ch = build_ch286(m)
    meta = {
        "hypothesis": "H_S5",
        "justification": (
            "Linear-dependence / L1 analysis: only tiny null space; drop 2 weakest "
            "L1 channels (288→286) with analytic in/out slicing of expand, DW, PW, "
            "and lateral. Prior H39 showed large width cuts fail — this tests whether "
            "a minimal cut is harmless (expected savings <<20%)."
        ),
        "layers_changed": [SP_EXPAND_W, *DW_W, *PW_W, LATERAL_W],
        "channel_slice": ch,
    }
    meta["params_before"] = base_params
    meta["params_after"] = param_count(m)
    records.append(save_candidate(m, "S4_ch286", meta))

    manifest = {
        "protocol": "structural-candidates-v1",
        "extension": "structural compression (not strict fixed-architecture WSP)",
        "baseline_sha256": file_sha256(BASE),
        "baseline_params": base_params,
        "kps_retained": True,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "negatives_not_built": [
            {
                "id": "BN_fusion",
                "reason": "0 BatchNormalization nodes in export; already folded",
            },
            {
                "id": "near_dup_filter_merge",
                "reason": "0 PW288 filter pairs with corr>0.99",
            },
            {
                "id": "PW_to_Identity",
                "reason": "rel_err_to_alpha_I≈1.0; prior H43 Hard~-23",
            },
        ],
        "candidates": records,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "structural_candidates_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(json.dumps({c["name"]: c["params"] for c in records}, indent=2))


if __name__ == "__main__":
    main()
