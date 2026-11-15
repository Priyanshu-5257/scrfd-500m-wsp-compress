#!/usr/bin/env python3
"""Phase-1 larger WIDER latency benchmark (no new compression).

Fixed threads, real WIDER images, bootstrap uncertainty on medians.
Preserves prior ablation/latency JSON; writes new phase1_* artifacts.
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

from eval.scrfd_onnx import SCRFD  # noqa: E402
from wsp.run_ablation_benchmark import estimate_macs  # noqa: E402
from wsp.run_latency_investigation import (  # noqa: E402
    INPUT_SIZE,
    THRESH,
    letterbox,
    make_session,
    summarize,
)

ABL = ROOT / "models" / "ablation"
OUT = ROOT / "wsp" / "analysis"
IMG_ROOT = ROOT / "data" / "widerface" / "WIDER_val" / "images"
DEV_SPLIT = ROOT / "wsp" / "splits" / "dev.json"

VARIANTS = [
    ("A_baseline", ABL / "A_baseline.onnx", "detect+landmarks"),
    ("B_kps_only", ABL / "B_kps_only.onnx", "detect_only"),
    ("C_svd_only", ABL / "C_svd_only.onnx", "detect+landmarks"),
    ("D_svd_nokps", ABL / "D_svd_nokps.onnx", "detect_only"),
]

THREADS = 4
N_IMAGES = int(os.environ.get("PHASE1_N_IMAGES", "96"))
N_ORT_PER_IMAGE = 3
N_E2E_PER_IMAGE = 2
N_BOOT = 1000
SEED = 0


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
    except Exception:
        info["cpu_model"] = "unavailable"
    return info


def load_images(n: int, seed: int) -> list[dict]:
    split = json.loads(DEV_SPLIT.read_text())
    ids = [im["id"] for im in split["images"]]
    rng = random.Random(seed)
    rng.shuffle(ids)
    out = []
    for iid in ids:
        path = IMG_ROOT / iid
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is None:
            continue
        out.append(
            {
                "id": iid,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "img": img,
            }
        )
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"Only loaded {len(out)}/{n} images")
    return out


def bootstrap_median_ci(samples: list[float], n_boot: int = N_BOOT, seed: int = 1):
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        meds.append(float(np.median(arr[idx])))
    meds = np.sort(meds)
    lo = float(meds[int(0.025 * (n_boot - 1))])
    hi = float(meds[int(0.975 * (n_boot - 1))])
    return {
        "median": float(np.median(arr)),
        "ci95_low": lo,
        "ci95_high": hi,
        "n_boot": n_boot,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_images(N_IMAGES, SEED)
    print(f"Loaded {len(images)} WIDER DEV images; threads={THREADS}", flush=True)

    # Precompute blobs
    blobs = []
    for item in images:
        det_img, _ = letterbox(item["img"])
        blob = cv2.dnn.blobFromImage(
            det_img, 1.0 / 128.0, INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True
        )
        blobs.append(blob)

    # Historical AP (do not re-eval unless already known)
    hist_ap = {
        "A_baseline": {"dev": (91.20, 87.27, 67.06), "source": "ablation_r081"},
        "B_kps_only": {"dev": (91.20, 87.27, 67.06), "source": "ablation_r082"},
        "C_svd_only": {"dev": (90.35, 86.80, 66.39), "source": "ablation_r083"},
        "D_svd_nokps": {"dev": (90.35, 86.80, 66.39), "source": "ablation_r084"},
    }

    rows = []
    order = [v[0] for v in VARIANTS]
    rng = random.Random(42)
    rng.shuffle(order)
    print("Variant order:", order, flush=True)

    path_by = {n: p for n, p, _ in VARIANTS}
    pipe_by = {n: pipe for n, _, pipe in VARIANTS}

    for name in order:
        path = path_by[name]
        print(f"=== {name} ===", flush=True)
        macs = estimate_macs(path, input_hw=INPUT_SIZE)
        params = sum(
            int(np.prod(x.dims))
            for x in __import__("onnx").load(str(path)).graph.initializer
            if "weight" in x.name or "bias" in x.name or True
        )
        # more accurate: from ablation manifest
        man = json.loads((ABL / "manifest.json").read_text())
        params = next(v["params"] for v in man["variants"] if v["name"] == name)

        sess = make_session(path, THREADS)
        iname = sess.get_inputs()[0].name
        outs = [o.name for o in sess.get_outputs()]
        # warmup
        for b in blobs[:3]:
            for _ in range(5):
                sess.run(outs, {iname: b})
        ort_times = []
        for b in blobs:
            for _ in range(N_ORT_PER_IMAGE):
                t0 = time.perf_counter()
                sess.run(outs, {iname: b})
                ort_times.append((time.perf_counter() - t0) * 1000.0)
        del sess

        det = SCRFD(str(path), providers=["CPUExecutionProvider"])
        det.session = make_session(path, THREADS)
        det._init_vars()
        e2e_times = []
        n_dets = []
        for item in images[: min(len(images), N_IMAGES)]:
            for _ in range(1):
                det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
            for _ in range(N_E2E_PER_IMAGE):
                t0 = time.perf_counter()
                boxes, _ = det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
                e2e_times.append((time.perf_counter() - t0) * 1000.0)
                n_dets.append(int(boxes.shape[0]))

        row = {
            "variant": name,
            "pipeline": pipe_by[name],
            "params": params,
            "bytes": path.stat().st_size,
            "gmacs": macs["gmacs"],
            "n_conv": macs["n_conv"],
            "threads": THREADS,
            "n_images": len(images),
            "ort_forward_ms": {
                **summarize(ort_times),
                "bootstrap_median_ci95": bootstrap_median_ci(ort_times, seed=11),
            },
            "detect_e2e_ms": {
                **summarize(e2e_times),
                "bootstrap_median_ci95": bootstrap_median_ci(e2e_times, seed=12),
            },
            "n_final_dets": summarize([float(x) for x in n_dets]),
            "wider_dev_ap_historical": {
                "easy": hist_ap[name]["dev"][0],
                "medium": hist_ap[name]["dev"][1],
                "hard": hist_ap[name]["dev"][2],
                "source": hist_ap[name]["source"],
                "note": "AP not re-measured in this latency-only Phase-1 run",
            },
        }
        rows.append(row)
        print(
            f"  ort_med={row['ort_forward_ms']['median_ms']:.2f} "
            f"ci=[{row['ort_forward_ms']['bootstrap_median_ci95']['ci95_low']:.2f},"
            f"{row['ort_forward_ms']['bootstrap_median_ci95']['ci95_high']:.2f}] "
            f"e2e_med={row['detect_e2e_ms']['median_ms']:.2f} "
            f"params={params} gmacs={macs['gmacs']:.3f}",
            flush=True,
        )

    # stable order in output
    rows.sort(key=lambda r: [v[0] for v in VARIANTS].index(r["variant"]))

    report = {
        "protocol": "phase1-latency-finalize-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": cpu_info(),
        "settings": {
            "threads_intra": THREADS,
            "threads_inter": 1,
            "input": "1x3x640x640",
            "n_images": len(images),
            "n_ort_per_image": N_ORT_PER_IMAGE,
            "n_e2e_per_image": N_E2E_PER_IMAGE,
            "score_thresh": THRESH,
            "provider": "CPUExecutionProvider",
            "image_ids": [im["id"] for im in images],
        },
        "historical_note": (
            "Preserves prior ablation AP/params; this run adds larger-N latency "
            "with bootstrap CI. ORT numbers supersede default-thread synthetic-blob "
            "medians for latency claims."
        ),
        "variants": rows,
    }
    outp = OUT / "phase1_latency_benchmark.json"
    outp.write_text(json.dumps(report, indent=2))
    print("Wrote", outp)

    # CSV
    lines = [
        "variant,pipeline,params,bytes,gmacs,ort_median_ms,ort_p95_ms,ort_ci95_low,ort_ci95_high,"
        "e2e_median_ms,e2e_p95_ms,e2e_ci95_low,e2e_ci95_high,dev_easy,dev_medium,dev_hard"
    ]
    for r in rows:
        o = r["ort_forward_ms"]
        e = r["detect_e2e_ms"]
        ap = r["wider_dev_ap_historical"]
        lines.append(
            f"{r['variant']},{r['pipeline']},{r['params']},{r['bytes']},{r['gmacs']:.6f},"
            f"{o['median_ms']:.4f},{o['p95_ms']:.4f},"
            f"{o['bootstrap_median_ci95']['ci95_low']:.4f},{o['bootstrap_median_ci95']['ci95_high']:.4f},"
            f"{e['median_ms']:.4f},{e['p95_ms']:.4f},"
            f"{e['bootstrap_median_ci95']['ci95_low']:.4f},{e['bootstrap_median_ci95']['ci95_high']:.4f},"
            f"{ap['easy']},{ap['medium']},{ap['hard']}"
        )
    csvp = ROOT / "PHASE1_LATENCY_BENCHMARK.csv"
    csvp.write_text("\n".join(lines) + "\n")
    print("Wrote", csvp)


if __name__ == "__main__":
    main()
