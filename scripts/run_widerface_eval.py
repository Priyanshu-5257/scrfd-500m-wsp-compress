#!/usr/bin/env python3
"""Run SCRFD-500M on WIDER Face val and report Easy/Medium/Hard AP."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scrfd_onnx import SCRFD
from eval.widerface_eval import evaluation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=str(ROOT / "models" / "det_500m.onnx"),
        help="SCRFD ONNX model path",
    )
    p.add_argument(
        "--images",
        default=str(ROOT / "data" / "widerface" / "WIDER_val" / "images"),
        help="WIDER Face val images root",
    )
    p.add_argument(
        "--gt",
        default=str(
            ROOT / "data" / "widerface" / "eval_tools" / "eval_tools" / "ground_truth"
        ),
        help="Directory with wider_{face,easy,medium,hard}_val.mat",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "widerface_preds"),
        help="Prediction txt output directory",
    )
    p.add_argument("--input-size", type=int, nargs=2, default=[640, 640], metavar=("W", "H"))
    p.add_argument("--score-thresh", type=float, default=0.02)
    p.add_argument("--nms-thresh", type=float, default=0.45)
    p.add_argument("--limit", type=int, default=0, help="Optional image limit for smoke tests")
    p.add_argument("--skip-infer", action="store_true", help="Reuse existing predictions")
    p.add_argument("--cpu", action="store_true", help="Force CPUExecutionProvider")
    return p.parse_args()


def list_images(images_root: Path):
    paths = sorted(images_root.glob("*/*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No images under {images_root}")
    return paths


def write_pred(out_root: Path, rel_event: str, stem: str, rel_name: str, boxes_xywh_score):
    event_dir = out_root / rel_event
    event_dir.mkdir(parents=True, exist_ok=True)
    out_file = event_dir / f"{stem}.txt"
    with open(out_file, "w") as f:
        f.write(f"{rel_name}\n")
        f.write(f"{boxes_xywh_score.shape[0]}\n")
        for b in boxes_xywh_score:
            f.write(f"{b[0]:.5f} {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")


def run_inference(args) -> dict:
    providers = ["CPUExecutionProvider"] if args.cpu else None
    detector = SCRFD(args.model, providers=providers)
    detector.nms_thresh = args.nms_thresh
    print(f"Providers: {detector.session.get_providers()}")
    print(f"use_kps={detector.use_kps} strides={detector._feat_stride_fpn}")

    images_root = Path(args.images)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    paths = list_images(images_root)
    if args.limit > 0:
        paths = paths[: args.limit]

    input_size = tuple(args.input_size)
    t0 = time.perf_counter()
    n_det = 0
    for path in tqdm(paths, desc="Infer"):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        dets, _ = detector.detect(img, thresh=args.score_thresh, input_size=input_size)
        # WIDER submission format: x, y, w, h, score
        if dets.shape[0]:
            xywh = dets.copy()
            xywh[:, 2] = dets[:, 2] - dets[:, 0]
            xywh[:, 3] = dets[:, 3] - dets[:, 1]
        else:
            xywh = dets
        n_det += int(xywh.shape[0])
        event = path.parent.name
        rel_name = f"{event}/{path.name}"
        write_pred(out_root, event, path.stem, rel_name, xywh)
    elapsed = time.perf_counter() - t0
    stats = {
        "num_images": len(paths),
        "num_detections": n_det,
        "infer_seconds": elapsed,
        "images_per_sec": len(paths) / elapsed if elapsed > 0 else 0.0,
        "input_size": list(input_size),
        "score_thresh": args.score_thresh,
        "nms_thresh": args.nms_thresh,
        "model": args.model,
        "providers": detector.session.get_providers(),
        "use_kps": bool(detector.use_kps),
    }
    return stats


def main():
    args = parse_args()
    for required in (args.model, args.images, args.gt):
        if not os.path.exists(required):
            raise FileNotFoundError(required)

    if not args.skip_infer:
        stats = run_inference(args)
    else:
        stats = {"skipped_infer": True}

    if args.limit > 0:
        print(f"Skipping official AP (--limit={args.limit}); inference smoke OK.")
        print(json.dumps(stats, indent=2))
        return

    # Published references (InsightFace README, VGA / official protocol)
    published = {
        "SCRFD_500M": {"easy": 90.57, "medium": 88.12, "hard": 68.51},
        "SCRFD_500M_KPS": {"easy": 90.97, "medium": 88.44, "hard": 69.49},
    }

    aps = evaluation(args.out, args.gt, iou_thresh=0.5)
    result = {
        **stats,
        "ap": {"easy": aps[0] * 100, "medium": aps[1] * 100, "hard": aps[2] * 100},
        "published_reference_percent": published,
        "note": (
            "buffalo_s/sc det_500m.onnx is the KPS variant; compare primarily to SCRFD_500M_KPS. "
            "This harness uses letterboxed input (default 640x640), score_thr=0.02, nms=0.45, IoU=0.5."
        ),
    }

    out_json = Path(args.out).parent / "widerface_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_json}")
    print(
        f"AP% Easy/Medium/Hard = "
        f"{result['ap']['easy']:.2f} / {result['ap']['medium']:.2f} / {result['ap']['hard']:.2f}"
    )
    print(
        "Published SCRFD_500M_KPS = "
        f"{published['SCRFD_500M_KPS']['easy']:.2f} / "
        f"{published['SCRFD_500M_KPS']['medium']:.2f} / "
        f"{published['SCRFD_500M_KPS']['hard']:.2f}"
    )


if __name__ == "__main__":
    main()
