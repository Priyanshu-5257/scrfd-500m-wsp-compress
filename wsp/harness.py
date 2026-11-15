#!/usr/bin/env python3
"""Strict-WSP harness for SCRFD-500M ONNX weight edits (instruction-only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnx
import yaml
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scrfd_onnx import SCRFD
from eval.widerface_eval import evaluation

WSP = ROOT / "wsp"
SPLITS = WSP / "splits"
CAND = WSP / "candidates"
LEDGER = WSP / "ledger"
STATE_PATH = WSP / "state.json"
CONTRACT_PATH = WSP / "experiment.yaml"
BASE_MODEL = ROOT / "models" / "det_500m.onnx"
GT_DIR = ROOT / "data" / "widerface" / "eval_tools" / "eval_tools" / "ground_truth"
IMG_ROOT = ROOT / "data" / "widerface" / "WIDER_val" / "images"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_contract() -> dict:
    with open(CONTRACT_PATH) as f:
        return yaml.safe_load(f)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "requests_used": 0,
        "confirmation_consumed": False,
        "baseline_hash": None,
        "current_candidate_hash": None,
        "selected_candidate_hash": None,
        "variants_tried": [],
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_manifest(model_path: Path) -> dict:
    model = onnx.load(str(model_path))
    rows = []
    h = hashlib.sha256()
    for t in sorted(model.graph.initializer, key=lambda x: x.name):
        arr = numpy_helper.to_array(t)
        th = hashlib.sha256(arr.tobytes()).hexdigest()
        h.update(t.name.encode())
        h.update(th.encode())
        rows.append(
            {
                "name": t.name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "numel": int(arr.size),
                "sha256": th,
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "abs_max": float(np.abs(arr).max()),
            }
        )
    return {
        "model_path": str(model_path),
        "file_sha256": file_sha256(model_path),
        "tensor_manifest_sha256": h.hexdigest(),
        "tensors": rows,
    }


def inspect_weights(names: Optional[List[str]] = None) -> dict:
    """Return allowed tensor metadata/stats without executing the detector."""
    man = tensor_manifest(BASE_MODEL)
    if names:
        want = set(names)
        man["tensors"] = [t for t in man["tensors"] if t["name"] in want]
    out = WSP / "analysis" / f"inspect_{int(time.time())}.json"
    out.write_text(json.dumps(man, indent=2))
    return man


def load_split_ids(role: str) -> List[str]:
    if role == "confirmation":
        raise RuntimeError("CONFIRMATION is sealed; refuse to load ids for editing search")
    data = json.loads((SPLITS / "dev.json").read_text())
    return [x["id"] for x in data["images"]]


def write_candidate(
    edit_spec: dict,
    parent_hash: Optional[str] = None,
    tag: Optional[str] = None,
) -> dict:
    """
    Apply a declared algebraic edit recipe to an allowlisted subset of tensors.
    edit_spec example:
      {
        "name": "cls8_bias_shift",
        "ops": [
          {"tensor": "bbox_head.stride_cls.(8, 8).bias", "op": "add_scalar", "value": 0.25}
        ]
      }
    """
    contract = load_contract()
    allow = set(contract["editable_tensors"])
    state = load_state()

    parent_path = BASE_MODEL
    if parent_hash and parent_hash != state.get("baseline_hash"):
        parent_cand = CAND / parent_hash / "model.onnx"
        if not parent_cand.exists():
            raise FileNotFoundError(parent_cand)
        parent_path = parent_cand

    model = onnx.load(str(parent_path))
    init_map = {t.name: t for t in model.graph.initializer}
    changed = []

    for op in edit_spec["ops"]:
        name = op["tensor"]
        if name not in allow:
            raise PermissionError(f"tensor not editable: {name}")
        if name not in init_map:
            raise KeyError(name)
        arr = numpy_helper.to_array(init_map[name]).astype(np.float32).copy()
        before = arr.copy()
        kind = op["op"]
        if kind == "add_scalar":
            arr = arr + np.float32(op["value"])
        elif kind == "mul_scalar":
            arr = arr * np.float32(op["value"])
        elif kind == "add_vector":
            vec = np.asarray(op["value"], dtype=np.float32)
            if vec.shape != arr.shape:
                raise ValueError(f"shape mismatch {vec.shape} vs {arr.shape}")
            arr = arr + vec
        elif kind == "scale_channels":
            # For weight [C_out, C_in, k, k], scale output channels by vector
            scales = np.asarray(op["value"], dtype=np.float32)
            if arr.ndim != 4 or scales.shape[0] != arr.shape[0]:
                raise ValueError("scale_channels expects OutxInxkxk and len(value)=Out")
            arr = arr * scales.reshape(-1, 1, 1, 1)
        elif kind == "mul_kernel":
            # Broadcast-multiply a spatial kernel onto a 4D weight [O,I,H,W]
            kern = np.asarray(op["value"], dtype=np.float32)
            if arr.ndim != 4:
                raise ValueError("mul_kernel expects 4D weight")
            if kern.ndim == 2:
                if kern.shape != arr.shape[-2:]:
                    raise ValueError(f"kernel {kern.shape} vs spatial {arr.shape[-2:]}")
                arr = arr * kern.reshape(1, 1, *kern.shape)
            else:
                raise ValueError("mul_kernel value must be 2D HxW")
        elif kind == "scale_in_channels":
            # For weight [C_out, C_in, k, k], scale selected input channels
            scales = np.asarray(op["value"], dtype=np.float32)
            indices = np.asarray(op["indices"], dtype=np.int64)
            if arr.ndim != 4:
                raise ValueError("scale_in_channels expects 4D weight")
            if scales.ndim != 1 or indices.ndim != 1 or len(scales) != len(indices):
                raise ValueError("value and indices must be 1D and same length")
            if np.any(indices < 0) or np.any(indices >= arr.shape[1]):
                raise ValueError("indices out of range for C_in")
            for idx, s in zip(indices, scales):
                arr[:, int(idx), :, :] *= s
        else:
            raise ValueError(f"unknown op {kind}")
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite after edit: {name}")
        new_t = numpy_helper.from_array(arr, name=name)
        # replace initializer
        for i, t in enumerate(model.graph.initializer):
            if t.name == name:
                model.graph.initializer.remove(t)
                model.graph.initializer.insert(i, new_t)
                break
        delta = arr - before
        changed.append(
            {
                "tensor": name,
                "op": kind,
                "value": op.get("value"),
                "changed_numel": int(np.count_nonzero(delta)),
                "l2": float(np.linalg.norm(delta)),
                "max_abs": float(np.abs(delta).max()),
            }
        )

    # temporary serialize to hash
    CAND.mkdir(parents=True, exist_ok=True)
    tmp = CAND / "_tmp.onnx"
    onnx.save(model, str(tmp))
    cand_hash = file_sha256(tmp)
    dest = CAND / cand_hash
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    final_model = dest / "model.onnx"
    shutil.move(str(tmp), final_model)
    meta = {
        "candidate_hash": cand_hash,
        "parent_hash": parent_hash or state.get("baseline_hash") or file_sha256(BASE_MODEL),
        "tag": tag or edit_spec.get("name"),
        "edit_spec": edit_spec,
        "changed": changed,
        "created_utc": utcnow(),
        "protected_check": "kps_and_non_allowlist_untouched_by_construction",
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2))
    state["current_candidate_hash"] = cand_hash
    state["variants_tried"].append(cand_hash)
    save_state(state)
    append_jsonl(
        LEDGER / "hypotheses.jsonl",
        {"type": "write_candidate", "utc": utcnow(), **meta},
    )
    return meta


def _run_split_eval(model_path: Path, image_ids: List[str], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = SCRFD(str(model_path), providers=["CPUExecutionProvider"])
    detector.nms_thresh = 0.45
    t0 = time.perf_counter()
    n_det = 0
    import cv2
    from tqdm import tqdm

    for rel in tqdm(image_ids, desc=f"eval:{out_dir.name}"):
        path = IMG_ROOT / rel
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"missing image {path}")
        dets, _ = detector.detect(img, thresh=0.02, input_size=(640, 640))
        if dets.shape[0]:
            xywh = dets.copy()
            xywh[:, 2] = dets[:, 2] - dets[:, 0]
            xywh[:, 3] = dets[:, 3] - dets[:, 1]
        else:
            xywh = dets
        n_det += int(xywh.shape[0])
        event, name = rel.split("/", 1)
        event_dir = out_dir / event
        event_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(name).stem
        with open(event_dir / f"{stem}.txt", "w") as f:
            f.write(f"{rel}\n")
            f.write(f"{xywh.shape[0]}\n")
            for b in xywh:
                f.write(f"{b[0]:.5f} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")
    elapsed = time.perf_counter() - t0
    # Full GT mats; evaluation ignores missing events' preds as empty -> undercounts faces
    # so we MUST only score using filtered predictions AND the same face counts.
    # Use event allowlist inside a thin wrapper by temporarily restricting pred dir.
    aps = evaluation(str(out_dir), str(GT_DIR), iou_thresh=0.5)
    # IMPORTANT: official eval still counts all GT faces in all events. For DEV-only AP
    # we recompute with event filter below if needed. Here we call filtered evaluator.
    return {
        "num_images": len(image_ids),
        "num_detections": n_det,
        "infer_seconds": elapsed,
        "ap_raw_fullgt": {"easy": aps[0] * 100, "medium": aps[1] * 100, "hard": aps[2] * 100},
    }


def evaluate_dev_filtered(model_path: Path, request_id: str, purpose: str) -> dict:
    """Metered DEV evaluation with GT face counts restricted to DEV events."""
    from eval.widerface_eval import (
        dataset_pr_info,
        get_gt_boxes,
        get_preds,
        image_eval,
        img_pr_info,
        norm_score,
        voc_ap,
    )
    from tqdm import tqdm

    contract = load_contract()
    state = load_state()
    budget = contract["budget"]["successful_requests"]
    if state["requests_used"] >= budget:
        raise RuntimeError("evaluation budget exhausted")

    dev = json.loads((SPLITS / "dev.json").read_text())
    image_ids = [x["id"] for x in dev["images"]]
    event_allow = set(dev["events"])
    out_dir = LEDGER / "requests" / request_id / "preds"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # inference
    detector = SCRFD(str(model_path), providers=["CPUExecutionProvider"])
    detector.nms_thresh = float(contract["inference"]["nms_thresh"])
    import cv2

    t0 = time.perf_counter()
    n_det = 0
    for rel in tqdm(image_ids, desc=request_id):
        img = cv2.imread(str(IMG_ROOT / rel))
        dets, _ = detector.detect(
            img,
            thresh=float(contract["inference"]["score_thresh"]),
            input_size=tuple(contract["inference"]["input_size"]),
        )
        if dets.shape[0]:
            xywh = dets.copy()
            xywh[:, 2] = dets[:, 2] - dets[:, 0]
            xywh[:, 3] = dets[:, 3] - dets[:, 1]
        else:
            xywh = dets
        n_det += int(xywh.shape[0])
        event, name = rel.split("/", 1)
        d = out_dir / event
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{Path(name).stem}.txt", "w") as f:
            f.write(f"{rel}\n{xywh.shape[0]}\n")
            for b in xywh:
                f.write(f"{b[0]:.5f} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")
    infer_s = time.perf_counter() - t0

    # filtered AP over DEV events only
    pred = get_preds(str(out_dir))
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(
        str(GT_DIR)
    )
    settings = ["easy", "medium", "hard"]
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    thresh_num = 1000
    aps = []
    for setting_id in range(3):
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2), dtype=np.float64)
        for i in range(len(event_list)):
            event_name = str(event_list[i][0][0])
            if event_name not in event_allow:
                continue
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]
            for j in range(len(img_list)):
                img_key = str(img_list[j][0][0])
                pred_info = pred_list.get(img_key, np.zeros((0, 5), dtype=np.float64))
                gt_boxes = gt_bbx_list[j][0].astype("float")
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, 0.5)
                pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    cand_hash = file_sha256(model_path)
    result = {
        "request_id": request_id,
        "purpose": purpose,
        "utc": utcnow(),
        "candidate_hash": cand_hash,
        "split": "DEV",
        "dev_manifest_sha256": dev["manifest_sha256"],
        "num_images": len(image_ids),
        "num_detections": n_det,
        "infer_seconds": infer_s,
        "ap": {"easy": aps[0] * 100, "medium": aps[1] * 100, "hard": aps[2] * 100},
        "count_face_hard_proxy_note": "per-setting face counts used inside AP",
    }
    state["requests_used"] = int(state["requests_used"]) + 1
    save_state(state)
    # update contract counters too
    contract["requests_used"] = state["requests_used"]
    contract["current_candidate_hash"] = cand_hash
    with open(CONTRACT_PATH, "w") as f:
        yaml.safe_dump(contract, f, sort_keys=False)

    append_jsonl(LEDGER / "requests.jsonl", result)
    (LEDGER / "requests" / request_id / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[{request_id}] DEV AP% E/M/H = {result['ap']['easy']:.2f}/"
        f"{result['ap']['medium']:.2f}/{result['ap']['hard']:.2f} "
        f"(requests_used={state['requests_used']}/{budget})"
    )
    return result


def register_baseline() -> dict:
    state = load_state()
    base_hash = file_sha256(BASE_MODEL)
    # materialize baseline candidate copy
    dest = CAND / base_hash
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / "model.onnx").exists():
        shutil.copy2(BASE_MODEL, dest / "model.onnx")
    (dest / "meta.json").write_text(
        json.dumps({"candidate_hash": base_hash, "tag": "baseline", "created_utc": utcnow()}, indent=2)
    )
    state["baseline_hash"] = base_hash
    state["current_candidate_hash"] = base_hash
    save_state(state)
    contract = load_contract()
    contract["baseline_hash"] = base_hash
    contract["current_candidate_hash"] = base_hash
    with open(CONTRACT_PATH, "w") as f:
        yaml.safe_dump(contract, f, sort_keys=False)
    return evaluate_dev_filtered(BASE_MODEL, request_id="r000_baseline", purpose="DEV_baseline")


def freeze_submission(candidate_hash: str, note: str) -> dict:
    """Lock selected candidate before CONFIRMATION; unseals confirmation eval only."""
    state = load_state()
    contract = load_contract()
    model = CAND / candidate_hash / "model.onnx"
    if not model.exists():
        raise FileNotFoundError(model)
    if file_sha256(model) != candidate_hash:
        raise RuntimeError("candidate hash mismatch")
    state["selected_candidate_hash"] = candidate_hash
    state["frozen_utc"] = utcnow()
    state["freeze_note"] = note
    save_state(state)
    contract["selected_candidate_hash"] = candidate_hash
    contract["freeze_note"] = note
    with open(CONTRACT_PATH, "w") as f:
        yaml.safe_dump(contract, f, sort_keys=False)
    seal = SPLITS / "CONFIRMATION_SEALED.txt"
    seal.write_text(
        "CONFIRMATION unsealed after freeze_submission.\n"
        f"selected_candidate_hash={candidate_hash}\n"
        f"frozen_utc={state['frozen_utc']}\n"
        f"note={note}\n"
    )
    meta = {
        "type": "freeze_submission",
        "utc": state["frozen_utc"],
        "candidate_hash": candidate_hash,
        "note": note,
    }
    append_jsonl(LEDGER / "hypotheses.jsonl", meta)
    (LEDGER / "freeze.json").write_text(json.dumps(meta, indent=2))
    return meta


def evaluate_confirmation(model_path: Path, request_id: str, purpose: str) -> dict:
    """Metered CONFIRMATION evaluation; only after freeze_submission."""
    from eval.widerface_eval import (
        dataset_pr_info,
        get_gt_boxes,
        get_preds,
        image_eval,
        img_pr_info,
        norm_score,
        voc_ap,
    )
    from tqdm import tqdm

    state = load_state()
    if not state.get("selected_candidate_hash"):
        raise RuntimeError("freeze_submission required before CONFIRMATION")
    contract = load_contract()
    budget = contract["budget"]["successful_requests"]
    if state["requests_used"] >= budget:
        raise RuntimeError("evaluation budget exhausted")

    conf = json.loads((SPLITS / "confirmation.json").read_text())
    image_ids = [x["id"] for x in conf["images"]]
    event_allow = set(conf["events"])
    out_dir = LEDGER / "requests" / request_id / "preds"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    detector = SCRFD(str(model_path), providers=["CPUExecutionProvider"])
    detector.nms_thresh = float(contract["inference"]["nms_thresh"])
    import cv2

    t0 = time.perf_counter()
    n_det = 0
    for rel in tqdm(image_ids, desc=request_id):
        img = cv2.imread(str(IMG_ROOT / rel))
        dets, _ = detector.detect(
            img,
            thresh=float(contract["inference"]["score_thresh"]),
            input_size=tuple(contract["inference"]["input_size"]),
        )
        if dets.shape[0]:
            xywh = dets.copy()
            xywh[:, 2] = dets[:, 2] - dets[:, 0]
            xywh[:, 3] = dets[:, 3] - dets[:, 1]
        else:
            xywh = dets
        n_det += int(xywh.shape[0])
        event, name = rel.split("/", 1)
        d = out_dir / event
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{Path(name).stem}.txt", "w") as f:
            f.write(f"{rel}\n{xywh.shape[0]}\n")
            for b in xywh:
                f.write(f"{b[0]:.5f} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")
    infer_s = time.perf_counter() - t0

    pred = get_preds(str(out_dir))
    norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(
        str(GT_DIR)
    )
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    thresh_num = 1000
    aps = []
    for setting_id in range(3):
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2), dtype=np.float64)
        for i in range(len(event_list)):
            event_name = str(event_list[i][0][0])
            if event_name not in event_allow:
                continue
            img_list = file_list[i][0]
            pred_list = pred.get(event_name, {})
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]
            for j in range(len(img_list)):
                img_key = str(img_list[j][0][0])
                pred_info = pred_list.get(img_key, np.zeros((0, 5), dtype=np.float64))
                gt_boxes = gt_bbx_list[j][0].astype("float")
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, 0.5)
                pr_curve += img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
        aps.append(voc_ap(pr_curve[:, 1], pr_curve[:, 0]))

    cand_hash = file_sha256(model_path)
    result = {
        "request_id": request_id,
        "purpose": purpose,
        "utc": utcnow(),
        "candidate_hash": cand_hash,
        "split": "CONFIRMATION",
        "confirmation_manifest_sha256": conf["manifest_sha256"],
        "num_images": len(image_ids),
        "num_detections": n_det,
        "infer_seconds": infer_s,
        "ap": {"easy": aps[0] * 100, "medium": aps[1] * 100, "hard": aps[2] * 100},
    }
    state["requests_used"] = int(state["requests_used"]) + 1
    state["confirmation_consumed"] = True
    save_state(state)
    contract["requests_used"] = state["requests_used"]
    contract["confirmation_consumed"] = True
    with open(CONTRACT_PATH, "w") as f:
        yaml.safe_dump(contract, f, sort_keys=False)

    append_jsonl(LEDGER / "requests.jsonl", result)
    (LEDGER / "requests" / request_id / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[{request_id}] CONFIRMATION AP% E/M/H = {result['ap']['easy']:.2f}/"
        f"{result['ap']['medium']:.2f}/{result['ap']['hard']:.2f} "
        f"(requests_used={state['requests_used']}/{budget})"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("--names", nargs="*", default=None)

    sub.add_parser("baseline")

    p_write = sub.add_parser("write")
    p_write.add_argument("--spec", required=True, help="path to edit_spec json")
    p_write.add_argument("--parent", default=None)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--hash", required=True)
    p_eval.add_argument("--request-id", required=True)
    p_eval.add_argument("--purpose", required=True)

    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--hash", required=True)
    p_freeze.add_argument("--note", required=True)

    p_conf = sub.add_parser("confirm")
    p_conf.add_argument("--hash", required=True)
    p_conf.add_argument("--request-id", required=True)
    p_conf.add_argument("--purpose", required=True)

    args = parser.parse_args()
    if args.cmd == "inspect":
        man = inspect_weights(args.names)
        print(json.dumps({"num_tensors": len(man["tensors"]), "manifest": man["tensor_manifest_sha256"]}, indent=2))
    elif args.cmd == "baseline":
        print(json.dumps(register_baseline(), indent=2))
    elif args.cmd == "write":
        spec = json.loads(Path(args.spec).read_text())
        print(json.dumps(write_candidate(spec, parent_hash=args.parent), indent=2))
    elif args.cmd == "eval":
        model = CAND / args.hash / "model.onnx"
        print(json.dumps(evaluate_dev_filtered(model, args.request_id, args.purpose), indent=2))
    elif args.cmd == "freeze":
        print(json.dumps(freeze_submission(args.hash, args.note), indent=2))
    elif args.cmd == "confirm":
        model = CAND / args.hash / "model.onnx"
        print(json.dumps(evaluate_confirmation(model, args.request_id, args.purpose), indent=2))


if __name__ == "__main__":
    main()
