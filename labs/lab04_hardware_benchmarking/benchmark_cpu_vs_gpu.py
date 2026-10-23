"""Comprehensive hardware benchmark suite comparing CPU vs Apple Silicon GPU (MPS) and generating comparison charts."""

from __future__ import annotations
import gc
import os
import statistics
import time
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import laya.presets as presets

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import print_header, print_metrics_table, get_hardware_profile, synchronize_device


def plot_hardware_benchmark_chart(results: Dict[str, Any], output_path: str = os.path.join(os.path.dirname(__file__), "hardware_benchmark_results.png")) -> str:
    """Generates a comprehensive 3-panel hardware performance visualization."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Laya Hardware Benchmark: Apple Silicon CPU vs GPU (MPS) on {results['hw']['cpu_brand']}", fontsize=14, fontweight="bold")

    devices = ["Apple CPU", "Apple GPU (MPS)"]
    colors = ["#4A90E2", "#50E3C2"]

    # Panel 1: Latency (p50 / p95)
    ax1 = axes[0]
    p50_vals = [results["single"]["cpu"]["p50"], results["single"]["mps"]["p50"]]
    p95_vals = [results["single"]["cpu"]["p95"], results["single"]["mps"]["p95"]]
    x = np.arange(len(devices))
    width = 0.35
    rects1 = ax1.bar(x - width/2, p50_vals, width, label="p50 Latency", color="#4A90E2")
    rects2 = ax1.bar(x + width/2, p95_vals, width, label="p95 Latency", color="#F5A623")
    ax1.set_title("Single-Request Latency (ms - lower is better)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(devices, fontweight="bold")
    ax1.set_ylabel("Latency (ms)")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for rect in rects1:
        y = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, y + 2, f"{y:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        y = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, y + 2, f"{y:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: Batch Throughput Scaling
    ax2 = axes[1]
    batch_sizes = results["batch"]["batch_sizes"]
    cpu_tps = [results["batch"]["cpu"][b]["throughput_rps"] for b in batch_sizes]
    mps_tps = [results["batch"]["mps"][b]["throughput_rps"] for b in batch_sizes]
    ax2.plot(batch_sizes, cpu_tps, marker="o", linewidth=2.5, color="#4A90E2", label="Apple CPU")
    ax2.plot(batch_sizes, mps_tps, marker="s", linewidth=2.5, color="#50E3C2", label="Apple GPU (MPS)")
    ax2.set_title("Throughput Scaling by Batch Size (higher is better)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Batch Size (Parallel Requests)")
    ax2.set_ylabel("Throughput (Requests / sec)")
    ax2.set_xticks(batch_sizes)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Multi-Head Question Scaling
    ax3 = axes[2]
    head_counts = results["heads"]["head_counts"]
    cpu_head_lats = [results["heads"]["cpu"][h] for h in head_counts]
    mps_head_lats = [results["heads"]["mps"][h] for h in head_counts]
    ax3.plot(head_counts, cpu_head_lats, marker="o", linewidth=2.5, color="#4A90E2", label="Apple CPU")
    ax3.plot(head_counts, mps_head_lats, marker="^", linewidth=2.5, color="#50E3C2", label="Apple GPU (MPS)")
    ax3.set_title("Multi-Head Question Scaling (ms)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Number of Parallel Question Heads")
    ax3.set_ylabel("Latency (ms)")
    ax3.set_xticks(head_counts)
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def run_full_hardware_benchmark(output_image_path: str = os.path.join(os.path.dirname(__file__), "hardware_benchmark_results.png")):
    hw = get_hardware_profile()
    
    print_header(
        "Lab 04: Comprehensive Laya Hardware Benchmark Suite",
        f"Device: {hw['cpu_brand']} ({hw['cpu_cores']} Cores) | Accelerator: {hw['gpu_name']} | PyTorch {hw['torch_version']}"
    )

    questions = presets.triage_questions()
    sample_text = (
        "Our primary PostgreSQL cluster is throwing connection timeout errors and customer "
        "checkout transactions are failing with HTTP 504 gateway timeouts."
    )

    import laya
    print("\n[1] Initializing Laya on CPU and Apple GPU (MPS)...")
    cpu_agent = laya.load("convaiinnovations/laya", device="cpu")
    mps_agent = laya.load("convaiinnovations/laya", device="mps")

    # 1. Single Request Latency Benchmark
    print("\n[2] Benchmarking Single Request Latency...")
    single_res = {"cpu": {}, "mps": {}}
    for dev, ag in [("cpu", cpu_agent), ("mps", mps_agent)]:
        # Warmup
        for _ in range(5):
            ag.predict(state={"message": sample_text}, questions=questions)
            synchronize_device(dev)

        lats = []
        for _ in range(25):
            synchronize_device(dev)
            t0 = time.perf_counter()
            ag.predict(state={"message": sample_text}, questions=questions)
            synchronize_device(dev)
            lats.append((time.perf_counter() - t0) * 1000.0)

        single_res[dev] = {
            "p50": float(np.percentile(lats, 50)),
            "p95": float(np.percentile(lats, 95)),
            "mean": float(statistics.mean(lats)),
            "throughput": 1000.0 / float(statistics.mean(lats)),
        }

    # 2. Batch Throughput Benchmark
    print("\n[3] Benchmarking Batch Scaling...")
    batch_sizes = [1, 4, 8, 16]
    batch_res = {"batch_sizes": batch_sizes, "cpu": {}, "mps": {}}
    for bsz in batch_sizes:
        batch_inputs = [{"message": f"{sample_text} [Req #{idx}]"} for idx in range(bsz)]
        for dev, ag in [("cpu", cpu_agent), ("mps", mps_agent)]:
            for _ in range(2):
                for item in batch_inputs:
                    ag.predict(state=item, questions=questions)
                synchronize_device(dev)

            t0 = time.perf_counter()
            for item in batch_inputs:
                ag.predict(state=item, questions=questions)
            synchronize_device(dev)
            elapsed = time.perf_counter() - t0
            tp = bsz / elapsed
            batch_res[dev][bsz] = {
                "throughput_rps": tp,
                "latency_per_req_ms": (elapsed / bsz) * 1000.0,
            }

    # 3. Question Head Scaling
    print("\n[4] Benchmarking Multi-Head Scaling...")
    head_counts = [1, 2, 4, 8]
    multi_questions = {
        "intent": {"type": "choice", "instructions": "Intent?", "criteria": {"tech": "tech", "bill": "bill"}},
        "is_outage": {"type": "noul", "instructions": "Outage?"},
        "severity": {"type": "score", "instructions": "Severity?", "criteria": ["p3", "p2", "p1"]},
        "is_churn": {"type": "noul", "instructions": "Churn risk?"},
        "department": {"type": "choice", "instructions": "Dept?", "criteria": {"sre": "sre", "fin": "fin"}},
        "priority": {"type": "score", "instructions": "Priority?", "criteria": ["low", "high"]},
        "callback": {"type": "noul", "instructions": "Callback needed?"},
        "security": {"type": "noul", "instructions": "Security issue?"},
    }
    head_res = {"head_counts": head_counts, "cpu": {}, "mps": {}}
    q_keys = list(multi_questions.keys())
    for cnt in head_counts:
        curr_q = {k: multi_questions[k] for k in q_keys[:cnt]}
        for dev, ag in [("cpu", cpu_agent), ("mps", mps_agent)]:
            for _ in range(2):
                ag.predict(state={"message": sample_text}, questions=curr_q)
                synchronize_device(dev)

            t0 = time.perf_counter()
            for _ in range(10):
                ag.predict(state={"message": sample_text}, questions=curr_q)
                synchronize_device(dev)
            head_res[dev][cnt] = ((time.perf_counter() - t0) / 10.0) * 1000.0

    full_results = {
        "hw": hw,
        "single": single_res,
        "batch": batch_res,
        "heads": head_res,
    }

    # Summary table
    headers = ["Metric / Benchmark", "Apple CPU", "Apple GPU (MPS)", "MPS Advantage"]
    rows = [
        ["Single p50 Latency", f"{single_res['cpu']['p50']:.1f} ms", f"{single_res['mps']['p50']:.1f} ms", f"🚀 {single_res['cpu']['p50']/single_res['mps']['p50']:.2f}x Faster"],
        ["Single p95 Latency", f"{single_res['cpu']['p95']:.1f} ms", f"{single_res['mps']['p95']:.1f} ms", f"🚀 {single_res['cpu']['p95']/single_res['mps']['p95']:.2f}x Faster"],
        ["Single Throughput", f"{single_res['cpu']['throughput']:.1f} req/s", f"{single_res['mps']['throughput']:.1f} req/s", f"🚀 {single_res['mps']['throughput']/single_res['cpu']['throughput']:.2f}x Higher"],
        ["Batch 16 Throughput", f"{batch_res['cpu'][16]['throughput_rps']:.1f} req/s", f"{batch_res['mps'][16]['throughput_rps']:.1f} req/s", f"🚀 {batch_res['mps'][16]['throughput_rps']/batch_res['cpu'][16]['throughput_rps']:.2f}x Higher"],
        ["8-Head Marginal Latency", f"{head_res['cpu'][8]:.1f} ms", f"{head_res['mps'][8]:.1f} ms", f"🚀 {head_res['cpu'][8]/head_res['mps'][8]:.2f}x Faster"],
    ]
    print_metrics_table(headers, rows, title=f"Apple M4 Max Hardware Performance Summary")

    img_path = plot_hardware_benchmark_chart(full_results, output_path=output_image_path)
    print(f"\n[5] Benchmark comparison chart saved to: {os.path.abspath(img_path)}")

    return full_results


if __name__ == "__main__":
    run_full_hardware_benchmark()
