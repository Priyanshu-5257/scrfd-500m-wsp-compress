#!/usr/bin/env python3
"""Build A/B/C/D ablation ONNX variants from immutable baseline (no training)."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import deque
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models" / "det_500m.onnx"
OUT = ROOT / "models" / "ablation"
ANALYSIS = ROOT / "wsp" / "analysis"

PW = [
    ("626", "628", "345", "346"),
    ("634", "636", "351", "352"),
    ("642", "644", "357", "358"),
    ("650", "652", "363", "364"),
    ("658", "660", "369", "370"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_count(model: onnx.ModelProto) -> int:
    return int(sum(numpy_helper.to_array(t).size for t in model.graph.initializer))


def component_params(model: onnx.ModelProto) -> dict:
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    kps = sum(a.size for n, a in init.items() if "kps" in n)
    head_other = sum(
        a.size
        for n, a in init.items()
        if n.startswith("bbox_head") and "kps" not in n
    )
    neck = sum(a.size for n, a in init.items() if n.startswith("neck"))
    # late 288 PW originals or factored
    pw_names = {w for w, *_ in PW} | {b for _, b, *_ in PW}
    svd_like = sum(
        a.size
        for n, a in init.items()
        if n.startswith("pw") and ("_A_r" in n or "_B_r" in n)
    )
    backbone = sum(a.size for n, a in init.items() if n.isdigit() or n in pw_names)
    # recount cleanly
    named_head_kps = kps
    named_head_det = head_other
    named_neck = neck
    other = 0
    backbone_total = 0
    for n, a in init.items():
        if "kps" in n:
            continue
        if n.startswith("bbox_head"):
            continue
        if n.startswith("neck"):
            continue
        if n.startswith("pw") and ("_A_r" in n or "_B_r" in n):
            continue
        if n.isdigit() or True:
            # remaining = backbone numeric + misc constants + svd factors counted separate
            if n.isdigit() or (len(n) <= 4 and n.replace(".", "").isdigit()):
                backbone_total += a.size
            elif n.startswith("pw"):
                continue
            else:
                # empty / scalar helpers
                other += a.size
    return {
        "total": param_count(model),
        "backbone_numeric": backbone_total,
        "svd_factors": svd_like,
        "neck": named_neck,
        "det_head_cls_reg": named_head_det,
        "kps_head": named_head_kps,
        "other_constants": other,
        # sanity: backbone + svd + neck + heads + other should ≈ total
        "sum_check": backbone_total
        + svd_like
        + named_neck
        + named_head_det
        + named_head_kps
        + other,
    }


def strip_kps(model: onnx.ModelProto) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    kps_w = {t.name for t in m.graph.initializer if "kps" in t.name}
    by_in: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            by_in.setdefault(i, []).append(n)
    reach: set[str] = set()
    q = deque(kps_w)
    while q:
        t = q.popleft()
        if t in reach:
            continue
        reach.add(t)
        for n in by_in.get(t, []):
            for o in n.output:
                if o not in reach:
                    q.append(o)
    kps_outs = {o.name for o in m.graph.output if o.name in reach}
    keep_outs = [o for o in m.graph.output if o.name not in kps_outs]
    del m.graph.output[:]
    m.graph.output.extend(keep_outs)

    prod = {o: n for n in m.graph.node for o in n.output}
    want = {o.name for o in m.graph.output}
    stack = list(want)
    keep_ids: set[int] = set()
    seen: set[str] = set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        n = prod.get(t)
        if not n:
            continue
        keep_ids.add(id(n))
        stack.extend(n.input)
    nodes = [n for n in m.graph.node if id(n) in keep_ids]
    del m.graph.node[:]
    m.graph.node.extend(nodes)
    used: set[str] = set()
    for n in m.graph.node:
        used.update(n.input)
    keep = [t for t in m.graph.initializer if t.name in used]
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    return m


def apply_svd(model: onnx.ModelProto, rank: int = 96) -> onnx.ModelProto:
    m = onnx.ModelProto()
    m.CopyFrom(model)
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    node_by_out = {o: n for n in m.graph.node for o in n.output}
    new_inits = []
    nodes_to_add = []
    for i, (wn, bn, din, dout) in enumerate(PW):
        W = init[wn].astype(np.float64).reshape(288, 288)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
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
    for n in m.graph.node:
        if n.op_type == "Conv" and len(n.output) == 1 and n.output[0] in pw_out_to_pair:
            new_nodes.extend(pair_nodes[pw_out_to_pair[n.output[0]]])
        else:
            new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    drop = {wn for wn, *_ in PW}
    keep = [t for t in m.graph.initializer if t.name not in drop]
    keep.extend(new_inits)
    del m.graph.initializer[:]
    m.graph.initializer.extend(keep)
    return m


def save_variant(model: onnx.ModelProto, name: str, description: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.onnx"
    onnx.save(model, str(path))
    try:
        onnx.checker.check_model(model)
        check_ok, check_err = True, None
    except Exception as e:  # noqa: BLE001
        check_ok, check_err = False, str(e)
    outs = [o.name for o in model.graph.output]
    meta = {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "description": description,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "params": param_count(model),
        "components": component_params(model),
        "num_outputs": len(outs),
        "outputs": outs,
        "has_kps_outputs": len(outs) == 9,
        "onnx_check_ok": check_ok,
        "onnx_check_err": check_err,
        "baseline_sha256": sha256(BASE),
    }
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    base = onnx.load(str(BASE))
    variants = []

    # A
    variants.append(
        save_variant(
            base,
            "A_baseline",
            "Original SCRFD-500M KPS checkpoint (immutable baseline copy)",
        )
    )
    # Also keep path alias copy if needed — same bytes as models/det_500m.onnx

    # B — KPS only
    variants.append(
        save_variant(
            strip_kps(base),
            "B_kps_only",
            "Baseline with keypoint head/outputs stripped; no SVD",
        )
    )

    # C — SVD only
    svd = apply_svd(base, rank=96)
    variants.append(
        save_variant(
            svd,
            "C_svd_only",
            "Rank-96 SVD factorization of five late PW layers; KPS retained",
        )
    )

    # D — Combined
    variants.append(
        save_variant(
            strip_kps(svd),
            "D_svd_nokps",
            "Rank-96 SVD + KPS strip (combined)",
        )
    )

    manifest = {
        "protocol": "ablation-svd-vs-kps-v1",
        "baseline_path": str(BASE.relative_to(ROOT)),
        "baseline_sha256": sha256(BASE),
        "rank": 96,
        "variants": variants,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "ablation_manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
