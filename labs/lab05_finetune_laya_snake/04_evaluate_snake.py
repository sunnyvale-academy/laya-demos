"""Evaluates and compares the Fine-Tuned Laya Snake Agent vs Random Policy & Base LLM."""

from __future__ import annotations
import os
import random
import statistics
import time
from typing import Dict, List, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import print_header, print_metrics_table, SnakeGame, Direction, get_hardware_profile
try:
    from labs.lab05_finetune_laya_snake.train_snake import SnakeFineTunedPolicy, extract_features_from_sample, REV_ACTION_MAP
except ImportError:
    from train_snake import SnakeFineTunedPolicy, extract_features_from_sample, REV_ACTION_MAP


def evaluate_policy(policy_name: str, model: Any = None, device: Any = None, num_episodes: int = 50, max_steps: int = 150) -> Dict[str, Any]:
    scores = []
    moves = []
    latencies_ms = []

    for ep in range(num_episodes):
        game = SnakeGame(width=10, height=10, seed=1000 + ep)
        step = 0

        while not game.game_over and step < max_steps:
            state_dict = game.get_state_dict()
            t0 = time.perf_counter()

            if policy_name == "finetuned_laya":
                sample = {
                    "state_text": state_dict["state_text"],
                    "metadata": {"snake_length": len(game.snake), "score": game.score}
                }
                feats = torch.tensor([extract_features_from_sample(sample)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    act_logits, _, _ = model(feats)
                    pred_idx = torch.argmax(act_logits, dim=-1).item()
                action = Direction(REV_ACTION_MAP[pred_idx])
            elif policy_name == "random":
                action = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
            elif policy_name == "expert_a_star":
                action = game.get_expert_action()

            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            game.step(action)
            step += 1

        scores.append(game.score)
        moves.append(game.steps_alive)

    return {
        "policy": policy_name,
        "avg_score": float(statistics.mean(scores)),
        "max_score": max(scores),
        "avg_moves": float(statistics.mean(moves)),
        "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
        "throughput_fps": 1000.0 / max(0.001, float(statistics.mean(latencies_ms))),
    }


def plot_snake_benchmark_chart(results: Dict[str, Any], output_path: str = "snake_benchmark_results.png") -> str:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Nokia 3310 Snake: Fine-Tuned Laya Decision Engine Benchmark", fontsize=14, fontweight="bold")

    policies = ["Random Baseline", "Fine-Tuned Laya", "Expert A*"]
    colors = ["#E74C3C", "#2ECC71", "#3498DB"]

    # 1. Average Score (Apples Eaten)
    avg_scores = [results["random"]["avg_score"], results["finetuned_laya"]["avg_score"], results["expert_a_star"]["avg_score"]]
    ax1 = axes[0]
    bars1 = ax1.bar(policies, avg_scores, color=colors, width=0.5)
    ax1.set_title("Average Score (Apples Eaten)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Apples Eaten per Game")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.3, f"{y:.1f}", ha="center", va="bottom", fontweight="bold")

    # 2. Decision Latency
    lats = [results["random"]["p50_latency_ms"], results["finetuned_laya"]["p50_latency_ms"], results["expert_a_star"]["p50_latency_ms"]]
    ax2 = axes[1]
    bars2 = ax2.bar(policies, lats, color=colors, width=0.5)
    ax2.set_title("Decision Latency (p50 ms - lower is better)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Milliseconds (ms)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        y = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, y + 0.05, f"{y:.2f}ms", ha="center", va="bottom", fontweight="bold")

    # 3. Decision Throughput (Moves / sec)
    tps = [results["random"]["throughput_fps"], results["finetuned_laya"]["throughput_fps"], results["expert_a_star"]["throughput_fps"]]
    ax3 = axes[2]
    bars3 = ax3.bar(policies, tps, color=colors, width=0.5)
    ax3.set_title("Throughput (Moves / sec - higher is better)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Decisions / sec (FPS)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        y = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, y + 10, f"{y:.0f}/s", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def run_snake_evaluation_suite(checkpoint_path: str = os.path.join(os.path.dirname(__file__), "snake_checkpoint.pt")):
    hw = get_hardware_profile()
    device_str = "mps" if hw["mps_available"] else ("cuda" if hw["cuda_available"] else "cpu")
    device = torch.device(device_str)

    print_header(
        "Lab 05.4: Evaluating Fine-Tuned Laya Snake Agent",
        f"Benchmarking Across 50 Simulated Match Episodes on {hw['cpu_brand']} ({device_str.upper()})"
    )

    model = SnakeFineTunedPolicy().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n[1] Evaluating Random Policy...")
    res_rand = evaluate_policy("random", num_episodes=50)

    print("[2] Evaluating Fine-Tuned Laya Policy...")
    res_laya = evaluate_policy("finetuned_laya", model=model, device=device, num_episodes=50)

    print("[3] Evaluating Expert A* Solver...")
    res_expert = evaluate_policy("expert_a_star", num_episodes=50)

    results = {
        "random": res_rand,
        "finetuned_laya": res_laya,
        "expert_a_star": res_expert,
    }

    # Summary table
    headers = ["Policy", "Avg Apples", "Max Apples", "Avg Survival (Moves)", "p50 Latency", "Throughput"]
    rows = [
        ["🎲 Random Policy", f"{res_rand['avg_score']:.1f}", f"{res_rand['max_score']}", f"{res_rand['avg_moves']:.1f}", f"{res_rand['p50_latency_ms']:.2f} ms", f"{res_rand['throughput_fps']:.0f} fps"],
        ["🧠 Fine-Tuned Laya", f"[bold green]{res_laya['avg_score']:.1f}[/bold green]", f"[bold green]{res_laya['max_score']}[/bold green]", f"[bold green]{res_laya['avg_moves']:.1f}[/bold green]", f"[bold cyan]{res_laya['p50_latency_ms']:.2f} ms[/bold cyan]", f"[bold cyan]{res_laya['throughput_fps']:.0f} fps[/bold cyan]"],
        ["⭐ Expert A* Solver", f"{res_expert['avg_score']:.1f}", f"{res_expert['max_score']}", f"{res_expert['avg_moves']:.1f}", f"{res_expert['p50_latency_ms']:.2f} ms", f"{res_expert['throughput_fps']:.0f} fps"],
    ]

    print_metrics_table(headers, rows, title="Snake Policy Benchmark Comparison")

    img_path = plot_snake_benchmark_chart(results, output_path=os.path.join(os.path.dirname(__file__), "snake_benchmark_results.png"))
    print(f"\n[4] Benchmark comparison chart saved to: {os.path.abspath(img_path)}")

    return results


if __name__ == "__main__":
    run_snake_evaluation_suite()
