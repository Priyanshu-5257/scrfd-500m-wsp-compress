#!/usr/bin/env python3
"""SVD low-rank factorization of SCRFD-500M late-backbone 288x288 PW convs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "models" / "det_500m.onnx"
CAND = ROOT / "wsp" / "candidates"
ANALYSIS = ROOT / "wsp" / "analysis"

PW = [
    ("626", "628", "345", "346"),
    ("634", "636", "351", "352"),
    ("642", "644", "357", "358"),
    ("650", "652", "363", "364"),
    ("658", "660", "369", "370"),
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_count(model: onnx.ModelProto) -> int:
    return int(sum(numpy_helper.to_array(t).size for t in model.graph.initializer))


def factor_pw(rank: int, tag: str | None = None) -> dict:
    model = onnx.load(str(BASE_MODEL))
    before = param_count(model)
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    node_by_out = {o: n for n in model.graph.node for o in n.output}

    new_inits = []
    nodes_to_add = []
    rel_errors = []

    for i, (wn, bn, din, dout) in enumerate(PW):
        W = init[wn].astype(np.float64).reshape(288, 288)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        Wr = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
        rel_errors.append(float(np.linalg.norm(W - Wr) / np.linalg.norm(W)))
        B = (np.diag(np.sqrt(S[:rank])) @ Vt[:rank, :]).astype(np.float32).reshape(
            rank, 288, 1, 1
        )
        A = (U[:, :rank] * np.sqrt(S[:rank])).astype(np.float32).reshape(288, rank, 1, 1)
        b_name = f"pw{i}_B_r{rank}"
        a_name = f"pw{i}_A_r{rank}"
        mid = f"pw{i}_mid_r{rank}"
        new_inits.append(numpy_helper.from_array(B, name=b_name))
        new_inits.append(numpy_helper.from_array(A, name=a_name))
        conv = node_by_out[dout]
        assert conv.op_type == "Conv" and wn in conv.input
        c1 = helper.make_node(
            "Conv",
            inputs=[din, b_name],
            outputs=[mid],
            name=f"pw{i}_down_r{rank}",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )
        c2 = helper.make_node(
            "Conv",
            inputs=[mid, a_name, bn],
            outputs=[dout],
            name=f"pw{i}_up_r{rank}",
            kernel_shape=[1, 1],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
            group=1,
        )
        nodes_to_add.extend([c1, c2])

    pw_out_to_pair = {dout: i for i, (*_, dout) in enumerate(PW)}
    pair_nodes = {i: (nodes_to_add[2 * i], nodes_to_add[2 * i + 1]) for i in range(5)}
    new_nodes = []
    for n in model.graph.node:
        if n.op_type == "Conv" and len(n.output) == 1 and n.output[0] in pw_out_to_pair:
            new_nodes.extend(pair_nodes[pw_out_to_pair[n.output[0]]])
        else:
            new_nodes.append(n)
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    drop = {wn for wn, *_ in PW}
    keep = [t for t in model.graph.initializer if t.name not in drop]
    keep.extend(new_inits)
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    after = param_count(model)
    try:
        onnx.checker.check_model(model)
        ok, err = True, None
    except Exception as e:  # noqa: BLE001
        ok, err = False, str(e)

    tag = tag or f"svd_r{rank}"
    meta = {
        "rank": rank,
        "pw_rel_frob_errors": rel_errors,
        "mean_rel_frob": float(np.mean(rel_errors)),
        "params_before": before,
        "params_after": after,
        "params_saved": before - after,
        "params_frac": after / before,
        "onnx_check_ok": ok,
        "onnx_check_err": err,
    }
    CAND.mkdir(parents=True, exist_ok=True)
    tmp = CAND / "_tmp_svd.onnx"
    onnx.save(model, str(tmp))
    cand_hash = file_sha256(tmp)
    dest = CAND / cand_hash
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.move(str(tmp), dest / "model.onnx")
    out = {
        "candidate_hash": cand_hash,
        "tag": tag,
        "experiment": "COMPRESS-svd-pw",
        "prune": meta,
    }
    (dest / "meta.json").write_text(json.dumps(out, indent=2))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / f"{tag}_meta.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    print(json.dumps(factor_pw(args.rank, args.tag), indent=2))


if __name__ == "__main__":
    main()
