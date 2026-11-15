#!/usr/bin/env python3
"""Structured width prune of SCRFD-500M late-backbone 288-ch stage (no training).

Separately declared COMPRESS experiment: architecture change via channel slicing.
Keeps top-L1 channels in the feedforward MobileNet 288 stage and rewrites ONNX
shapes + depthwise group attrs. Neck lateral_convs.2 input dim follows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "models" / "det_500m.onnx"
CAND = ROOT / "wsp" / "candidates"
ANALYSIS = ROOT / "wsp" / "analysis"

# Late-backbone 288 stage initializers (see graph dump).
EXPAND_W, EXPAND_B = "618", "620"
DW_W = ["622", "630", "638", "646", "654"]
DW_B = ["624", "632", "640", "648", "656"]
PW_W = ["626", "634", "642", "650", "658"]
PW_B = ["628", "636", "644", "652", "660"]
LATERAL_W = "neck.lateral_convs.2.conv.weight"
LATERAL_B = "neck.lateral_convs.2.conv.bias"
STAGE_BIAS = [
    EXPAND_B,
    *DW_B,
    *PW_B,
]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_count(model: onnx.ModelProto) -> int:
    return int(sum(numpy_helper.to_array(t).size for t in model.graph.initializer))


def channel_scores(init: dict) -> np.ndarray:
    """L1 importance over expand-out / DW / PW in+out / lateral-in."""
    score = np.abs(init[EXPAND_W]).sum(axis=(1, 2, 3)).astype(np.float64)
    for tid in PW_W:
        W = init[tid]
        score += np.abs(W).sum(axis=(1, 2, 3))
        score += np.abs(W).sum(axis=(0, 2, 3))
    for tid in DW_W:
        W = init[tid]
        score += np.abs(W).reshape(W.shape[0], -1).sum(1)
    for tid in STAGE_BIAS:
        score += np.abs(init[tid])
    Wl = init[LATERAL_W]
    score += np.abs(Wl).sum(axis=(0, 2, 3))
    return score


def keep_indices(score: np.ndarray, keep_c: int) -> np.ndarray:
    if keep_c <= 0 or keep_c > score.shape[0]:
        raise ValueError(f"keep_c={keep_c} invalid for C={score.shape[0]}")
    # keep highest scores; sort ascending for stable ONNX slicing order
    top = np.argsort(score)[-keep_c:]
    return np.sort(top)


def replace_initializer(model: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    new_t = numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name=name)
    for i, t in enumerate(model.graph.initializer):
        if t.name == name:
            model.graph.initializer.remove(t)
            model.graph.initializer.insert(i, new_t)
            return
    raise KeyError(name)


def set_conv_group(model: onnx.ModelProto, weight_name: str, group: int) -> None:
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        if weight_name not in node.input:
            continue
        found = False
        for attr in node.attribute:
            if attr.name == "group":
                attr.i = int(group)
                found = True
                break
        if not found:
            from onnx import helper

            node.attribute.append(helper.make_attribute("group", int(group)))
        return
    raise KeyError(f"Conv using weight {weight_name}")


def prune_model(keep_c: int, base_path: Path = BASE_MODEL) -> Tuple[onnx.ModelProto, dict]:
    model = onnx.load(str(base_path))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    C0 = init[EXPAND_W].shape[0]
    if C0 != 288:
        raise RuntimeError(f"expected 288-ch stage, got {C0}")

    score = channel_scores(init)
    idx = keep_indices(score, keep_c)
    meta = {
        "op": "structured_width_prune_288",
        "C0": C0,
        "keep_c": keep_c,
        "keep_indices": idx.tolist(),
        "score_method": "l1_expand_dw_pw_lateral",
        "dropped_score_sum": float(score.sum() - score[idx].sum()),
        "kept_score_frac": float(score[idx].sum() / score.sum()),
    }

    # Expand PW: (C,152,1,1)
    replace_initializer(model, EXPAND_W, init[EXPAND_W][idx])
    replace_initializer(model, EXPAND_B, init[EXPAND_B][idx])

    for w, b in zip(DW_W, DW_B):
        replace_initializer(model, w, init[w][idx])
        replace_initializer(model, b, init[b][idx])
        set_conv_group(model, w, keep_c)

    for w, b in zip(PW_W, PW_B):
        # (C_out, C_in, 1, 1) — uniform width: same keep set on in and out
        replace_initializer(model, w, init[w][idx][:, idx])
        replace_initializer(model, b, init[b][idx])

    # Lateral: (16, 288, 1, 1) → (16, keep_c, 1, 1)
    replace_initializer(model, LATERAL_W, init[LATERAL_W][:, idx])
    # bias unchanged shape

    # Drop stale value_info that still claims 288-ch activations.
    stale = {
        "340", "342", "343", "345", "346", "348", "349", "351", "352", "354",
        "355", "357", "358", "360", "361", "363", "364", "366", "367", "369",
        "370", "372",
    }
    keep_vi = [vi for vi in model.graph.value_info if vi.name not in stale]
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep_vi)

    meta["params_before"] = param_count(onnx.load(str(base_path)))
    meta["params_after"] = param_count(model)
    meta["params_saved"] = meta["params_before"] - meta["params_after"]
    meta["params_frac"] = meta["params_after"] / meta["params_before"]
    return model, meta


def write_pruned_candidate(keep_c: int, tag: str | None = None) -> dict:
    model, prune_meta = prune_model(keep_c)
    CAND.mkdir(parents=True, exist_ok=True)
    tmp = CAND / "_tmp_prune.onnx"
    onnx.save(model, str(tmp))
    # quick shape check via onnx checker (may warn on IR)
    try:
        onnx.checker.check_model(model)
        check_ok = True
        check_err = None
    except Exception as e:  # noqa: BLE001 — record checker issues
        check_ok = False
        check_err = str(e)

    cand_hash = file_sha256(tmp)
    dest = CAND / cand_hash
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    final = dest / "model.onnx"
    shutil.move(str(tmp), final)

    meta = {
        "candidate_hash": cand_hash,
        "parent_hash": file_sha256(BASE_MODEL),
        "tag": tag or f"prune288to{keep_c}",
        "experiment": "COMPRESS-structured-width",
        "architecture_change": True,
        "prune": prune_meta,
        "onnx_check_ok": check_ok,
        "onnx_check_err": check_err,
        "created_utc": utcnow(),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / f"prune288_C{keep_c}_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def dump_ranking() -> dict:
    model = onnx.load(str(BASE_MODEL))
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    score = channel_scores(init)
    order = np.argsort(score)
    out = {
        "C": int(score.shape[0]),
        "score_min": float(score.min()),
        "score_max": float(score.max()),
        "score_median": float(np.median(score)),
        "ascending_indices": order.tolist(),
        "ascending_scores": score[order].tolist(),
        "projected_params": {},
    }
    base_n = param_count(model)
    for c in [288, 256, 224, 192, 160, 144, 128, 96]:
        # analytical projection matching prune_model
        saved = 0
        # expand out + bias
        saved += (288 - c) * 152 + (288 - c)
        for _ in DW_W:
            saved += (288 - c) * 9 + (288 - c)
        for _ in PW_W:
            saved += 288 * 288 - c * c + (288 - c)
        saved += 16 * (288 - c)  # lateral in
        out["projected_params"][str(c)] = {
            "params": base_n - saved,
            "frac": (base_n - saved) / base_n,
            "saved": saved,
        }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (ANALYSIS / "prune288_channel_ranking.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rank")
    p = sub.add_parser("write")
    p.add_argument("--keep-c", type=int, required=True)
    p.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.cmd == "rank":
        print(json.dumps(dump_ranking(), indent=2)[:2000])
    else:
        print(json.dumps(write_pruned_candidate(args.keep_c, args.tag), indent=2))


if __name__ == "__main__":
    main()
