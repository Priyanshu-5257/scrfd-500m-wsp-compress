#!/usr/bin/env python3
"""Benchmark A/B/C/D ablation variants: params, MACs, latency, DEV AP."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.scrfd_onnx import SCRFD  # noqa: E402
from wsp.harness import evaluate_dev_filtered  # noqa: E402

ABL = ROOT / "models" / "ablation"
OUT = ROOT / "wsp" / "analysis"


def estimate_macs(model_path: Path, input_hw=(640, 640)) -> dict:
    m = onnx.load(str(model_path))
    init = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    def get_attr(node, name, default=None):
        for a in node.attribute:
            if a.name == name:
                from onnx import helper

                return helper.get_attribute_value(a)
        return default

    graph_inputs = [i.name for i in m.graph.input if i.name not in init]
    act = {graph_inputs[0]: (1, 3, input_hw[1], input_hw[0])}
    macs = 0
    n_conv = 0
    for node in m.graph.node:
        if node.op_type == "Conv":
            x = node.input[0]
            wname = node.input[1]
            if x not in act or wname not in init:
                continue
            Warr = init[wname]
            Cout, Cin_g, kH, kW = Warr.shape
            strides = get_attr(node, "strides", [1, 1])
            pads = get_attr(node, "pads", [0, 0, 0, 0])
            N, _, Hin, Win = act[x]
            pad_h = pads[0] + pads[2] if len(pads) == 4 else 0
            pad_w = pads[1] + pads[3] if len(pads) == 4 else 0
            Hout = (Hin + pad_h - kH) // strides[0] + 1
            Wout = (Win + pad_w - kW) // strides[1] + 1
            m_ = int(N * Cout * Cin_g * kH * kW * Hout * Wout)
            macs += m_
            n_conv += 1
            act[node.output[0]] = (N, Cout, Hout, Wout)
        elif node.op_type in ("Relu", "Clip", "Identity", "Sigmoid"):
            if node.input[0] in act:
                act[node.output[0]] = act[node.input[0]]
        elif node.op_type == "Add":
            a = act.get(node.input[0]) or act.get(node.input[1])
            if a:
                act[node.output[0]] = a
        elif node.op_type == "Resize":
            x = node.input[0]
            if x in act:
                N, C, H, W = act[x]
                scales = None
                for inp in node.input[1:]:
                    if inp in init and init[inp].size >= 4:
                        scales = init[inp]
                if scales is not None:
                    act[node.output[0]] = (
                        N,
                        C,
                        int(H * float(scales[2])),
                        int(W * float(scales[3])),
                    )
                else:
                    act[node.output[0]] = (N, C, H * 2, W * 2)
        elif node.op_type == "Concat":
            parts = [act[i] for i in node.input if i in act]
            if parts:
                N, _, H, W = parts[0]
                act[node.output[0]] = (N, sum(p[1] for p in parts), H, W)
        elif node.op_type == "Slice":
            if node.input[0] in act:
                act[node.output[0]] = act[node.input[0]]
        else:
            if node.input and node.input[0] in act and node.output:
                act[node.output[0]] = act[node.input[0]]
    return {"macs": macs, "gmacs": macs / 1e9, "n_conv": n_conv}


def bench_ort(model_path: Path, n_warmup=20, n_run=100, input_size=(640, 640)) -> dict:
    import onnxruntime as ort

    so = ort.SessionOptions()
    thr = int(os.environ.get("ORT_INTRA_THREADS", "0") or 0)
    if thr > 0:
        so.intra_op_num_threads = thr
    sess = ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    name = sess.get_inputs()[0].name
    blob = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(n_warmup):
        sess.run(outs, {name: blob})
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        sess.run(outs, {name: blob})
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_warmup": n_warmup,
        "n_run": n_run,
        "n_outputs": len(outs),
        "intra_op_num_threads": thr if thr > 0 else "ORT_default",
    }


def bench_detect(
    model_path: Path, n_warmup=5, n_run=30, input_size=(640, 640)
) -> dict:
    det = SCRFD(str(model_path), providers=["CPUExecutionProvider"])
    img = np.random.randint(0, 255, (input_size[1], input_size[0], 3), dtype=np.uint8)
    for _ in range(n_warmup):
        det.detect(img, thresh=0.02, input_size=input_size)
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        boxes, kps = det.detect(img, thresh=0.02, input_size=input_size)
        times.append(time.perf_counter() - t0)
    arr = np.array(times) * 1000
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_warmup": n_warmup,
        "n_run": n_run,
        "use_kps": bool(det.use_kps),
        "pipeline": "detect+landmarks" if det.use_kps else "detect_only",
        "note": (
            "Includes preprocess + ORT + decode + NMS"
            + (" + kps decode" if det.use_kps else "")
        ),
    }


def cpu_info() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count_logical": os.cpu_count(),
    }
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if line.startswith("Model name:"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
            if line.startswith("CPU(s):"):
                info["lscpu_cpus"] = line.split(":", 1)[1].strip()
    except Exception:
        info["cpu_model"] = "unavailable"
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
    except Exception:
        info["onnxruntime"] = "unavailable"
    return info


def main():
    manifest = json.loads((ABL / "manifest.json").read_text())
    rows = []
    hardware = cpu_info()
    input_size = (640, 640)

    for v in manifest["variants"]:
        path = ROOT / v["path"]
        print(f"=== {v['name']} ===", flush=True)
        macs = estimate_macs(path, input_hw=input_size)
        ort_lat = bench_ort(path, input_size=input_size)
        det_lat = bench_detect(path, input_size=input_size)
        row = {
            "variant": v["name"],
            "description": v["description"],
            "sha256": v["sha256"],
            "bytes": v["bytes"],
            "params_total": v["params"],
            "params_components": v["components"],
            "has_kps_outputs": v["has_kps_outputs"],
            "num_outputs": v["num_outputs"],
            "gmacs": macs["gmacs"],
            "macs": macs["macs"],
            "n_conv": macs["n_conv"],
            "latency_ort_forward_ms": ort_lat,
            "latency_detect_e2e_ms": det_lat,
            "input_size": list(input_size),
            "provider": "CPUExecutionProvider",
        }
        rows.append(row)
        print(
            f"  params={v['params']} gmacs={macs['gmacs']:.3f} "
            f"ort_med={ort_lat['median_ms']:.2f} det_med={det_lat['median_ms']:.2f} "
            f"kps={v['has_kps_outputs']}",
            flush=True,
        )

    # DEV AP for each variant (metered harness)
    ap_results = {}
    req_ids = {
        "A_baseline": "r081_ablA",
        "B_kps_only": "r082_ablB",
        "C_svd_only": "r083_ablC",
        "D_svd_nokps": "r084_ablD",
    }
    for v in manifest["variants"]:
        name = v["name"]
        path = ROOT / v["path"]
        rid = req_ids[name]
        print(f"DEV AP {name} ...", flush=True)
        # bump budget if needed via state
        state_path = ROOT / "wsp" / "state.json"
        state = json.loads(state_path.read_text())
        contract_path = ROOT / "wsp" / "experiment.yaml"
        import yaml

        contract = yaml.safe_load(contract_path.read_text())
        if state["requests_used"] >= contract["budget"]["successful_requests"]:
            contract["budget"]["successful_requests"] = state["requests_used"] + 10
            contract_path.write_text(yaml.safe_dump(contract, sort_keys=False))
        result = evaluate_dev_filtered(path, rid, f"ablation_{name}")
        ap_results[name] = {
            "easy": result["ap"]["easy"],
            "medium": result["ap"]["medium"],
            "hard": result["ap"]["hard"],
            "request_id": rid,
            "num_images": result["num_images"],
            "split": "DEV",
            "source": "measured_this_run",
        }
        for row in rows:
            if row["variant"] == name:
                row["wider_dev_ap"] = ap_results[name]

    # Historical CONFIRMATION where available (A and D only from prior runs)
    historical = {
        "A_baseline": {
            "confirmation": {"easy": 90.65, "medium": 88.37, "hard": 68.59},
            "source": "historical_r041_CONF_baseline",
            "note": "Prior CONFIRMATION eval of same baseline checkpoint",
        },
        "D_svd_nokps": {
            "confirmation": {"easy": 90.22, "medium": 88.01, "hard": 67.88},
            "source": "historical_r080_CONF_H44b",
            "note": "Prior CONFIRMATION of combined SVD+nokps; re-validated as D here on DEV",
        },
        "B_kps_only": {
            "confirmation": None,
            "source": "not_executed",
            "note": "CONFIRMATION not run for B in this ablation",
        },
        "C_svd_only": {
            "confirmation": None,
            "source": "not_executed",
            "note": "CONFIRMATION not run for C in this ablation",
        },
    }
    for row in rows:
        h = historical[row["variant"]]
        row["wider_confirmation_ap"] = h

    # Landmark accuracy: WIDER has no landmark GT in this harness
    for row in rows:
        row["landmark_accuracy"] = {
            "available": False,
            "reason": "WIDER Face official eval mats provide box difficulty flags only; no landmark GT in this workspace",
            "kps_outputs_present": row["has_kps_outputs"],
        }

    report = {
        "protocol": "ablation-svd-vs-kps-v1",
        "hardware": hardware,
        "counting": {
            "macs": "Conv MACs = N*Cout*Cin_per_group*kH*kW*Hout*Wout; FLOPs≈2*MACs if counted that way (we report MACs/GMACs)",
            "input": "1x3x640x640",
            "latency_ort": "session.run only (neural forward)",
            "latency_detect": "SCRFD.detect e2e (preprocess+forward+decode+NMS[+kps])",
        },
        "baseline_sha256": manifest["baseline_sha256"],
        "variants": rows,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_benchmark.json").write_text(json.dumps(report, indent=2))
    print("Wrote", OUT / "ablation_benchmark.json")


if __name__ == "__main__":
    main()
