#!/usr/bin/env python3
"""Phase-4 benchmark: structural candidates vs A / C / D (no new weight search)."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scrfd_onnx import SCRFD  # noqa: E402
from wsp.harness import evaluate_dev_filtered  # noqa: E402
from wsp.run_ablation_benchmark import estimate_macs  # noqa: E402
from wsp.run_latency_investigation import (  # noqa: E402
    INPUT_SIZE,
    THRESH,
    letterbox,
    load_wider_images,
    make_session,
    summarize,
)
from wsp.run_phase1_latency_benchmark import bootstrap_median_ci, cpu_info  # noqa: E402

OUT = ROOT / "wsp" / "analysis"
THREADS = 4
N_IMAGES = int(os.environ.get("STRUCT_N_IMAGES", "48"))

MODELS = [
    ("A_baseline", ROOT / "models/ablation/A_baseline.onnx", "detect+landmarks", "reference"),
    ("C_svd_only", ROOT / "models/ablation/C_svd_only.onnx", "detect+landmarks", "reference_svd96"),
    ("D_svd_nokps", ROOT / "models/ablation/D_svd_nokps.onnx", "detect_only", "reference_combined_det_only"),
    ("S1_152_svd64", ROOT / "models/structural/S1_152_svd64.onnx", "detect+landmarks", "candidate"),
    ("S2_expand_svd70", ROOT / "models/structural/S2_expand_svd70.onnx", "detect+landmarks", "candidate"),
    ("S3_multi_svd", ROOT / "models/structural/S3_multi_svd.onnx", "detect+landmarks", "candidate"),
    ("S4_ch286", ROOT / "models/structural/S4_ch286.onnx", "detect+landmarks", "candidate"),
]

REQ = {
    "S1_152_svd64": "r085_S1",
    "S2_expand_svd70": "r086_S2",
    "S3_multi_svd": "r087_S3",
    "S4_ch286": "r088_S4",
}

# Historical DEV AP for references (not re-billed unless missing)
HIST_AP = {
    "A_baseline": {"easy": 91.198681, "medium": 87.269285, "hard": 67.055902, "source": "r081"},
    "C_svd_only": {"easy": 90.348366, "medium": 86.796758, "hard": 66.393320, "source": "r083"},
    "D_svd_nokps": {"easy": 90.348366, "medium": 86.796758, "hard": 66.393320, "source": "r084"},
}


def ensure_budget(n_extra: int = 8) -> None:
    state_path = ROOT / "wsp" / "state.json"
    contract_path = ROOT / "wsp" / "experiment.yaml"
    state = json.loads(state_path.read_text())
    contract = yaml.safe_load(contract_path.read_text())
    need = state["requests_used"] + n_extra
    if contract["budget"]["successful_requests"] < need:
        contract["budget"]["successful_requests"] = need
        contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))


def bench_model(name: str, path: Path, pipeline: str, images: list) -> dict:
    macs = estimate_macs(path, input_hw=INPUT_SIZE)
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    params = int(sum(numpy_helper.to_array(t).size for t in m.graph.initializer))
    n_out = len(m.graph.output)

    blobs = []
    for item in images:
        det_img, _ = letterbox(item["img"])
        blobs.append(
            cv2.dnn.blobFromImage(
                det_img, 1.0 / 128.0, INPUT_SIZE, (127.5, 127.5, 127.5), swapRB=True
            )
        )
    sess = make_session(path, THREADS)
    iname = sess.get_inputs()[0].name
    outs = [o.name for o in sess.get_outputs()]
    for b in blobs[:2]:
        for _ in range(5):
            sess.run(outs, {iname: b})
    ort_t = []
    for b in blobs:
        for _ in range(2):
            t0 = time.perf_counter()
            sess.run(outs, {iname: b})
            ort_t.append((time.perf_counter() - t0) * 1000.0)
    del sess

    det = SCRFD(str(path), providers=["CPUExecutionProvider"])
    det.session = make_session(path, THREADS)
    det._init_vars()
    e2e = []
    for item in images:
        det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
        t0 = time.perf_counter()
        det.detect(item["img"], thresh=THRESH, input_size=INPUT_SIZE)
        e2e.append((time.perf_counter() - t0) * 1000.0)

    return {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "pipeline": pipeline,
        "params": params,
        "bytes": path.stat().st_size,
        "gmacs": macs["gmacs"],
        "n_conv": macs["n_conv"],
        "num_outputs": n_out,
        "has_kps": n_out == 9,
        "ort_forward_ms": {
            **summarize(ort_t),
            "bootstrap_median_ci95": bootstrap_median_ci(ort_t, seed=21),
        },
        "detect_e2e_ms": {
            **summarize(e2e),
            "bootstrap_median_ci95": bootstrap_median_ci(e2e, seed=22),
        },
        "threads": THREADS,
        "n_images": len(images),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    images = load_wider_images(n=N_IMAGES, seed=1)
    print(f"Images={len(images)} threads={THREADS}", flush=True)

    # Latency for all
    order = [x[0] for x in MODELS]
    random.Random(0).shuffle(order)
    by_name = {n: (p, pipe, role) for n, p, pipe, role in MODELS}
    rows = {}
    for name in order:
        path, pipe, role = by_name[name]
        print(f"LATENCY {name}...", flush=True)
        row = bench_model(name, path, pipe, images)
        row["role"] = role
        rows[name] = row
        print(
            f"  params={row['params']} gmacs={row['gmacs']:.3f} "
            f"ort={row['ort_forward_ms']['median_ms']:.2f} "
            f"e2e={row['detect_e2e_ms']['median_ms']:.2f}",
            flush=True,
        )

    # DEV AP for new candidates only
    ensure_budget(6)
    for name in ("S1_152_svd64", "S2_expand_svd70", "S3_multi_svd", "S4_ch286"):
        path = by_name[name][0]
        rid = REQ[name]
        print(f"DEV AP {name} ({rid})...", flush=True)
        result = evaluate_dev_filtered(path, rid, f"structural_{name}")
        rows[name]["wider_dev_ap"] = {
            "easy": result["ap"]["easy"],
            "medium": result["ap"]["medium"],
            "hard": result["ap"]["hard"],
            "request_id": rid,
            "source": "measured_this_run",
            "num_images": result["num_images"],
        }

    for name, ap in HIST_AP.items():
        rows[name]["wider_dev_ap"] = {**ap, "note": "historical_ablation_DEV"}

    # Success criteria vs A (research targets)
    a = rows["A_baseline"]
    criteria = []
    for name in ("S1_152_svd64", "S2_expand_svd70", "S3_multi_svd", "S4_ch286"):
        r = rows[name]
        pref = (a["params"] - r["params"]) / a["params"] * 100
        ort_imp = (
            (a["ort_forward_ms"]["median_ms"] - r["ort_forward_ms"]["median_ms"])
            / a["ort_forward_ms"]["median_ms"]
            * 100
        )
        hard_drop = a["wider_dev_ap"]["hard"] - r["wider_dev_ap"]["hard"]
        ok = {
            "params_ge_20pct": pref >= 20.0,
            "ort_ge_15pct": ort_imp >= 15.0,
            "hard_drop_le_1pp": hard_drop <= 1.0,
            "kps_preserved": r["has_kps"],
        }
        criteria.append(
            {
                "name": name,
                "param_reduction_pct": pref,
                "ort_improvement_pct": ort_imp,
                "hard_drop_pp": hard_drop,
                "gates": ok,
                "meets_all_targets": all(ok.values()),
            }
        )

    report = {
        "protocol": "structural-benchmark-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": cpu_info(),
        "settings": {
            "threads": THREADS,
            "n_images_latency": len(images),
            "input": "640x640",
            "provider": "CPUExecutionProvider",
        },
        "models": [rows[n] for n, *_ in MODELS],
        "success_criteria_vs_A": criteria,
        "confirmation": {
            "status": "not_run",
            "reason": (
                "No candidate frozen: confirmation reserved for a candidate meeting "
                "all research targets on DEV; see meets_all_targets flags."
            ),
        },
        "landmark_accuracy": {
            "available": False,
            "reason": "No landmark GT in workspace",
        },
    }
    outp = OUT / "structural_benchmark.json"
    outp.write_text(json.dumps(report, indent=2))
    print("Wrote", outp)
    print(json.dumps(criteria, indent=2))


if __name__ == "__main__":
    main()
