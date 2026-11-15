#!/usr/bin/env python3
"""Investigate A/B/C/D latency discrepancy (no weight edits).

Measures:
  1) ORT forward with fixed threads {1,2,4,8}, repeated trials, shuffled order
  2) Stage profile: preprocess / ORT / box-decode / kps-decode / NMS / total
  3) Candidate counts: anchors, post-thresh, post-NMS
  4) Real WIDER FACE images (same set for all variants)
  5) Controlled postprocess: equalize candidate budget and re-time

Preserves existing ablation artifacts; writes new analysis only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scrfd_onnx import SCRFD, distance2bbox, distance2kps  # noqa: E402

ABL = ROOT / "models" / "ablation"
OUT = ROOT / "wsp" / "analysis"
IMG_ROOT = ROOT / "data" / "widerface" / "WIDER_val" / "images"
DEV_SPLIT = ROOT / "wsp" / "splits" / "dev.json"

VARIANTS = [
    ("A_baseline", ABL / "A_baseline.onnx"),
    ("B_kps_only", ABL / "B_kps_only.onnx"),
    ("C_svd_only", ABL / "C_svd_only.onnx"),
    ("D_svd_nokps", ABL / "D_svd_nokps.onnx"),
]

THRESH = 0.02
INPUT_SIZE = (640, 640)
NMS_THRESH = 0.45


def cpu_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
        "onnxruntime": ort.__version__,
    }
    try:
        out = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("CPU(s):"):
                info["lscpu_cpus"] = line.split(":", 1)[1].strip()
    except Exception:
        info["cpu_model"] = "unavailable"
    return info


def load_wider_images(n: int = 24, seed: int = 0) -> list[dict]:
    split = json.loads(DEV_SPLIT.read_text())
    ids = [im["id"] for im in split["images"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    chosen = []
    for iid in ids:
        path = IMG_ROOT / iid
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        chosen.append(
            {
                "id": iid,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "h": int(img.shape[0]),
                "w": int(img.shape[1]),
                "img": img,
            }
        )
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"Only found {len(chosen)}/{n} WIDER images under {IMG_ROOT}")
    return chosen


def letterbox(img: np.ndarray, input_size=INPUT_SIZE):
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]
    resized = cv2.resize(img, (new_width, new_height))
    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized
    return det_img, det_scale


def make_session(model_path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def summarize(times_ms: list[float]) -> dict:
    arr = np.asarray(times_ms, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()) if arr.size else float("nan"),
        "std_ms": float(arr.std(ddof=0)) if arr.size else float("nan"),
        "median_ms": float(np.median(arr)) if arr.size else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if arr.size else float("nan"),
        "min_ms": float(arr.min()) if arr.size else float("nan"),
        "max_ms": float(arr.max()) if arr.size else float("nan"),
    }


def bench_ort_threads(
    images: list[dict],
    threads_list=(1, 2, 4, 8),
    n_trials: int = 3,
    n_warmup: int = 5,
    n_run_per_image: int = 5,
    seed: int = 42,
) -> dict:
    """ORT session.run only; shuffled variant order each trial."""
    rng = random.Random(seed)
    # Precompute blobs once (same inputs for all variants)
    blobs = []
    for item in images:
        det_img, _ = letterbox(item["img"])
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True
        )
        blobs.append(blob)

    results = {}
    for thr in threads_list:
        print(f"[ORT] threads={thr}", flush=True)
        # Create sessions once per thread setting
        sessions = {}
        for name, path in VARIANTS:
            sess = make_session(path, thr)
            sessions[name] = {
                "sess": sess,
                "input": sess.get_inputs()[0].name,
                "outputs": [o.name for o in sess.get_outputs()],
            }

        trial_medians = {name: [] for name, _ in VARIANTS}
        all_samples = {name: [] for name, _ in VARIANTS}

        for trial in range(n_trials):
            order = [name for name, _ in VARIANTS]
            rng.shuffle(order)
            print(f"  trial {trial+1}/{n_trials} order={order}", flush=True)
            for name in order:
                s = sessions[name]
                # warmup
                for b in blobs[: min(2, len(blobs))]:
                    for _ in range(n_warmup):
                        s["sess"].run(s["outputs"], {s["input"]: b})
                times = []
                for b in blobs:
                    for _ in range(n_run_per_image):
                        t0 = time.perf_counter()
                        s["sess"].run(s["outputs"], {s["input"]: b})
                        times.append((time.perf_counter() - t0) * 1000.0)
                all_samples[name].extend(times)
                trial_medians[name].append(float(np.median(times)))

        results[str(thr)] = {
            "intra_op_num_threads": thr,
            "inter_op_num_threads": 1,
            "n_trials": n_trials,
            "n_images": len(blobs),
            "n_run_per_image": n_run_per_image,
            "per_variant": {
                name: {
                    **summarize(all_samples[name]),
                    "trial_medians_ms": trial_medians[name],
                    "trial_median_of_medians_ms": float(np.median(trial_medians[name])),
                    "n_outputs": len(sessions[name]["outputs"]),
                }
                for name, _ in VARIANTS
            },
        }
        # free sessions
        del sessions
    return results


def nms_keep(dets: np.ndarray, thresh: float = NMS_THRESH) -> list:
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def profile_forward_stages(
    sess: ort.InferenceSession,
    use_kps: bool,
    det_img: np.ndarray,
    det_scale: float,
    thresh: float = THRESH,
    center_cache: dict | None = None,
    candidate_cap: int | None = None,
) -> dict:
    """Profile preprocess-ready forward: blob + ORT + decode + NMS.

    If candidate_cap is set, keep only the top-`candidate_cap` scores before NMS
    (and before stacking KPS for NMS-kept path), to equalize postprocess load.
    """
    if center_cache is None:
        center_cache = {}
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    fmc = 3
    feat_strides = [8, 16, 32]
    num_anchors = 2

    t_blob0 = time.perf_counter()
    blob = cv2.dnn.blobFromImage(
        det_img, 1.0 / 128.0, INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True
    )
    t_blob = (time.perf_counter() - t_blob0) * 1000.0

    t_ort0 = time.perf_counter()
    net_outs = sess.run(output_names, {input_name: blob})
    t_ort = (time.perf_counter() - t_ort0) * 1000.0

    input_height = blob.shape[2]
    input_width = blob.shape[3]
    scores_list, bboxes_list, kpss_list = [], [], []
    n_anchors_total = 0
    t_box = 0.0
    t_kps = 0.0
    t_thresh = 0.0

    for idx, stride in enumerate(feat_strides):
        scores = net_outs[idx]
        bbox_preds = net_outs[idx + fmc]
        if len(scores.shape) == 3:
            scores = scores[0]
            bbox_preds = bbox_preds[0]
        bbox_preds = bbox_preds * stride
        kps_preds = None
        if use_kps:
            kps_preds = net_outs[idx + fmc * 2]
            if len(kps_preds.shape) == 3:
                kps_preds = kps_preds[0]
            kps_preds = kps_preds * stride

        height = input_height // stride
        width = input_width // stride
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(
                np.float32
            )
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape(
                    (-1, 2)
                )
            center_cache[key] = anchor_centers
        n_anchors_total += int(anchor_centers.shape[0])

        t0 = time.perf_counter()
        pos_inds = np.where(scores.ravel() >= thresh)[0]
        t_thresh += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        t_box += (time.perf_counter() - t0) * 1000.0

        scores_list.append(scores.ravel()[pos_inds])
        bboxes_list.append(bboxes[pos_inds])

        if use_kps:
            t0 = time.perf_counter()
            kpss = distance2kps(anchor_centers, kps_preds).reshape(
                (anchor_centers.shape[0], -1, 2)
            )
            t_kps += (time.perf_counter() - t0) * 1000.0
            kpss_list.append(kpss[pos_inds])

    if len(scores_list) == 0 or all(len(s) == 0 for s in scores_list):
        return {
            "blob_ms": t_blob,
            "ort_ms": t_ort,
            "thresh_ms": t_thresh,
            "box_decode_ms": t_box,
            "kps_decode_ms": t_kps,
            "nms_ms": 0.0,
            "assemble_ms": 0.0,
            "total_post_ms": t_thresh + t_box + t_kps,
            "n_anchors": n_anchors_total,
            "n_post_thresh": 0,
            "n_post_thresh_used": 0,
            "n_post_nms": 0,
            "candidate_cap": candidate_cap,
            "use_kps": use_kps,
        }

    t_as0 = time.perf_counter()
    scores = np.hstack(scores_list)
    order = scores.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    n_post_thresh = int(pre_det.shape[0])

    if candidate_cap is not None and pre_det.shape[0] > candidate_cap:
        pre_det = pre_det[:candidate_cap]
        order_used = order[:candidate_cap]
    else:
        order_used = order
    n_used = int(pre_det.shape[0])
    t_assemble = (time.perf_counter() - t_as0) * 1000.0

    t_nms0 = time.perf_counter()
    keep = nms_keep(pre_det, NMS_THRESH)
    t_nms = (time.perf_counter() - t_nms0) * 1000.0
    n_post_nms = len(keep)

    # KPS gather (if present) — cheap relative to full decode, timed into kps
    if use_kps and len(kpss_list):
        t0 = time.perf_counter()
        kpss = np.vstack(kpss_list) / det_scale
        _ = kpss[order_used, :, :][keep, :, :]
        t_kps += (time.perf_counter() - t0) * 1000.0

    return {
        "blob_ms": t_blob,
        "ort_ms": t_ort,
        "thresh_ms": t_thresh,
        "box_decode_ms": t_box,
        "kps_decode_ms": t_kps,
        "nms_ms": t_nms,
        "assemble_ms": t_assemble,
        "total_post_ms": t_thresh + t_box + t_kps + t_assemble + t_nms,
        "n_anchors": n_anchors_total,
        "n_post_thresh": n_post_thresh,
        "n_post_thresh_used": n_used,
        "n_post_nms": n_post_nms,
        "candidate_cap": candidate_cap,
        "use_kps": use_kps,
    }


def stage_profile_suite(
    images: list[dict],
    threads: int = 4,
    n_warmup: int = 2,
    n_repeats: int = 3,
    seed: int = 7,
) -> dict:
    rng = random.Random(seed)
    order = [name for name, _ in VARIANTS]
    rng.shuffle(order)

    # Prepare letterboxed images once
    prepared = []
    for item in images:
        det_img, det_scale = letterbox(item["img"])
        prepared.append((item["id"], det_img, det_scale))

    # Also time letterbox preprocess separately (shared, image-dependent)
    preprocess_times = []
    for item in images:
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            letterbox(item["img"])
            preprocess_times.append((time.perf_counter() - t0) * 1000.0)

    out = {
        "threads": threads,
        "order": order,
        "preprocess_letterbox_ms": summarize(preprocess_times),
        "variants": {},
    }

    for name in order:
        path = dict(VARIANTS)[name]
        sess = make_session(path, threads)
        use_kps = len(sess.get_outputs()) == 9
        cache: dict = {}

        # warmup
        for _, det_img, det_scale in prepared[:2]:
            for _ in range(n_warmup):
                profile_forward_stages(sess, use_kps, det_img, det_scale, center_cache=cache)

        samples = []
        for _, det_img, det_scale in prepared:
            for _ in range(n_repeats):
                samples.append(
                    profile_forward_stages(
                        sess, use_kps, det_img, det_scale, center_cache=cache
                    )
                )

        def col(key):
            return [s[key] for s in samples]

        out["variants"][name] = {
            "use_kps": use_kps,
            "n_samples": len(samples),
            "stages_ms": {
                "blob": summarize(col("blob_ms")),
                "ort": summarize(col("ort_ms")),
                "thresh": summarize(col("thresh_ms")),
                "box_decode": summarize(col("box_decode_ms")),
                "kps_decode": summarize(col("kps_decode_ms")),
                "assemble": summarize(col("assemble_ms")),
                "nms": summarize(col("nms_ms")),
                "total_post": summarize(col("total_post_ms")),
                "total_infer_plus_post": summarize(
                    [a + b for a, b in zip(col("ort_ms"), col("total_post_ms"))]
                ),
                "total_with_blob": summarize(
                    [
                        a + b + c
                        for a, b, c in zip(
                            col("blob_ms"), col("ort_ms"), col("total_post_ms")
                        )
                    ]
                ),
            },
            "counts": {
                "n_anchors_mean": float(np.mean(col("n_anchors"))),
                "n_post_thresh": summarize([float(x) for x in col("n_post_thresh")]),
                "n_post_nms": summarize([float(x) for x in col("n_post_nms")]),
                "per_image_mean_post_thresh": {},  # filled below
            },
            "per_image": [],
        }

        # Aggregate per-image means for counts
        by_img = {}
        idx = 0
        for iid, det_img, det_scale in prepared:
            reps = samples[idx : idx + n_repeats]
            idx += n_repeats
            pt = [s["n_post_thresh"] for s in reps]
            pn = [s["n_post_nms"] for s in reps]
            nms_ms = [s["nms_ms"] for s in reps]
            ort_ms = [s["ort_ms"] for s in reps]
            post_ms = [s["total_post_ms"] for s in reps]
            by_img[iid] = {
                "n_post_thresh_mean": float(np.mean(pt)),
                "n_post_nms_mean": float(np.mean(pn)),
                "nms_ms_mean": float(np.mean(nms_ms)),
                "ort_ms_mean": float(np.mean(ort_ms)),
                "post_ms_mean": float(np.mean(post_ms)),
            }
        out["variants"][name]["per_image"] = [
            {"id": iid, **stats} for iid, stats in by_img.items()
        ]
        out["variants"][name]["counts"]["per_image_mean_post_thresh"] = {
            iid: by_img[iid]["n_post_thresh_mean"] for iid in by_img
        }
        print(
            f"[STAGE] {name}: ort_med={out['variants'][name]['stages_ms']['ort']['median_ms']:.2f} "
            f"nms_med={out['variants'][name]['stages_ms']['nms']['median_ms']:.2f} "
            f"post_thresh_med={out['variants'][name]['counts']['n_post_thresh']['median_ms']:.0f} "
            f"(count)",
            flush=True,
        )
        del sess
    return out


def controlled_candidate_cap_experiment(
    images: list[dict],
    threads: int = 4,
    caps=(50, 100, 200, 500, None),
    n_repeats: int = 3,
) -> dict:
    """Re-time postprocess with equalized top-K candidates; test if SVD e2e gap remains."""
    prepared = []
    for item in images:
        det_img, det_scale = letterbox(item["img"])
        prepared.append((item["id"], det_img, det_scale))

    results = {"threads": threads, "caps": {}, "note": (
        "candidate_cap=None means natural post-thresh count; "
        "finite caps keep only top-K scores before NMS"
    )}

    for cap in caps:
        cap_key = "natural" if cap is None else str(cap)
        print(f"[CAP] candidate_cap={cap_key}", flush=True)
        results["caps"][cap_key] = {}
        for name, path in VARIANTS:
            sess = make_session(path, threads)
            use_kps = len(sess.get_outputs()) == 9
            cache: dict = {}
            # warmup
            for _, det_img, det_scale in prepared[:2]:
                profile_forward_stages(
                    sess, use_kps, det_img, det_scale, center_cache=cache, candidate_cap=cap
                )
            samples = []
            for _, det_img, det_scale in prepared:
                for _ in range(n_repeats):
                    samples.append(
                        profile_forward_stages(
                            sess,
                            use_kps,
                            det_img,
                            det_scale,
                            center_cache=cache,
                            candidate_cap=cap,
                        )
                    )

            def col(k):
                return [s[k] for s in samples]

            results["caps"][cap_key][name] = {
                "use_kps": use_kps,
                "ort_ms": summarize(col("ort_ms")),
                "nms_ms": summarize(col("nms_ms")),
                "total_post_ms": summarize(col("total_post_ms")),
                "ort_plus_post_ms": summarize(
                    [a + b for a, b in zip(col("ort_ms"), col("total_post_ms"))]
                ),
                "n_post_thresh_natural": summarize(
                    [float(x) for x in col("n_post_thresh")]
                ),
                "n_post_thresh_used": summarize(
                    [float(x) for x in col("n_post_thresh_used")]
                ),
                "n_post_nms": summarize([float(x) for x in col("n_post_nms")]),
            }
            v = results["caps"][cap_key][name]
            print(
                f"  {name}: ort+post_med={v['ort_plus_post_ms']['median_ms']:.2f} "
                f"nms_med={v['nms_ms']['median_ms']:.2f} "
                f"used_cand_med={v['n_post_thresh_used']['median_ms']:.0f}",
                flush=True,
            )
            del sess
    return results


def e2e_scrfd_detect(images: list[dict], threads: int = 4, n_repeats: int = 3) -> dict:
    """Full SCRFD.detect wall time on real images (for comparison with prior ablation)."""
    out = {}
    for name, path in VARIANTS:
        # SCRFD class doesn't expose thread control; set via session recreation workaround
        # by temporarily monkeypatching: create detector then replace session
        det = SCRFD(str(path), providers=["CPUExecutionProvider"])
        det.session = make_session(path, threads)
        det._init_vars()
        times = []
        n_dets = []
        for item in images:
            for _ in range(max(1, n_repeats // 2)):
                det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                boxes, _ = det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
                times.append((time.perf_counter() - t0) * 1000.0)
                n_dets.append(int(boxes.shape[0]))
        out[name] = {
            "detect_e2e_ms": summarize(times),
            "n_final_dets": summarize([float(x) for x in n_dets]),
            "use_kps": bool(det.use_kps),
            "threads": threads,
        }
        print(
            f"[E2E] {name}: med={out[name]['detect_e2e_ms']['median_ms']:.2f} "
            f"dets_med={out[name]['n_final_dets']['median_ms']:.1f}",
            flush=True,
        )
    return out


def count_correlation(stage_profile: dict) -> dict:
    """Correlate n_post_thresh with nms_ms across images/variants."""
    rows = []
    for name, v in stage_profile["variants"].items():
        for pi in v["per_image"]:
            rows.append(
                {
                    "variant": name,
                    "id": pi["id"],
                    "n_post_thresh": pi["n_post_thresh_mean"],
                    "n_post_nms": pi["n_post_nms_mean"],
                    "nms_ms": pi["nms_ms_mean"],
                    "ort_ms": pi["ort_ms_mean"],
                    "post_ms": pi["post_ms_mean"],
                }
            )
    if not rows:
        return {}
    x = np.array([r["n_post_thresh"] for r in rows], dtype=np.float64)
    y = np.array([r["nms_ms"] for r in rows], dtype=np.float64)
    if x.std() < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
    # Per-variant mean counts
    by_var = {}
    for name in dict(VARIANTS):
        sub = [r for r in rows if r["variant"] == name]
        by_var[name] = {
            "mean_n_post_thresh": float(np.mean([r["n_post_thresh"] for r in sub])),
            "mean_n_post_nms": float(np.mean([r["n_post_nms"] for r in sub])),
            "mean_nms_ms": float(np.mean([r["nms_ms"] for r in sub])),
            "mean_ort_ms": float(np.mean([r["ort_ms"] for r in sub])),
            "mean_post_ms": float(np.mean([r["post_ms"] for r in sub])),
        }
    return {
        "pearson_r_n_post_thresh_vs_nms_ms": corr,
        "n_pairs": len(rows),
        "by_variant_means": by_var,
        "rows": rows,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_images = int(os.environ.get("LAT_N_IMAGES", "24"))
    print(f"Loading {n_images} WIDER DEV images...", flush=True)
    images = load_wider_images(n=n_images, seed=0)
    meta = {
        "protocol": "latency-investigation-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": cpu_info(),
        "input_size": list(INPUT_SIZE),
        "thresh": THRESH,
        "nms_thresh": NMS_THRESH,
        "n_images": len(images),
        "image_ids": [im["id"] for im in images],
        "image_sha256": {im["id"]: im["sha256"] for im in images},
        "variants": {n: str(p.relative_to(ROOT)) for n, p in VARIANTS},
        "preserves_prior": [
            "wsp/analysis/ablation_benchmark.json",
            "ABLATION_REPORT.md",
            "BENCHMARK_RESULTS.csv",
        ],
        "no_weight_edits": True,
    }

    print("=== 1) ORT thread sweep ===", flush=True)
    ort_threads = bench_ort_threads(
        images,
        threads_list=(1, 2, 4, 8),
        n_trials=3,
        n_warmup=3,
        n_run_per_image=3,
        seed=42,
    )

    print("=== 2/3/4) Stage profile + counts on real images ===", flush=True)
    stages = stage_profile_suite(images, threads=4, n_warmup=2, n_repeats=3, seed=7)

    print("=== 5) Controlled candidate-cap experiment ===", flush=True)
    caps = controlled_candidate_cap_experiment(
        images, threads=4, caps=(50, 100, 200, 500, None), n_repeats=3
    )

    print("=== E2E SCRFD.detect on real images ===", flush=True)
    e2e = e2e_scrfd_detect(images, threads=4, n_repeats=3)

    corr = count_correlation(stages)

    report = {
        **meta,
        "ort_thread_sweep": ort_threads,
        "stage_profile": stages,
        "controlled_candidate_caps": caps,
        "e2e_detect_real_images": e2e,
        "count_latency_correlation": {
            k: v for k, v in corr.items() if k != "rows"
        },
        "count_latency_rows": corr.get("rows", []),
    }
    out_path = OUT / "latency_investigation.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("Wrote", out_path, flush=True)

    # Compact CSV for ORT medians
    csv_lines = ["threads,variant,median_ms,p95_ms,mean_ms,trial_median_of_medians_ms"]
    for thr, block in ort_threads.items():
        for name, stats in block["per_variant"].items():
            csv_lines.append(
                f"{thr},{name},{stats['median_ms']:.4f},{stats['p95_ms']:.4f},"
                f"{stats['mean_ms']:.4f},{stats['trial_median_of_medians_ms']:.4f}"
            )
    csv_path = OUT / "latency_ort_threads.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print("Wrote", csv_path, flush=True)


if __name__ == "__main__":
    main()
