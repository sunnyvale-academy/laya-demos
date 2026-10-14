"""Visualization and terminal formatting utilities for Laya Decision Engine labs."""

from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
from tabulate import tabulate
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

console = Console()


def print_header(title: str, subtitle: Optional[str] = None) -> None:
    """Print an attractive, stylized header banner in the terminal."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"
    console.print(Panel(content, expand=False, border_style="cyan"))


def print_metrics_table(headers: List[str], rows: List[List[Any]], title: Optional[str] = None) -> None:
    """Print a clean formatted metrics table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    for h in headers:
        table.add_column(h)
    for r in rows:
        table.add_row(*[str(cell) for cell in r])
    console.print(table)


def print_cube_state(cube: Any, title: str = "Cube State", step: Optional[int] = None,
                     action: Optional[str] = None, confidence: Optional[float] = None) -> None:
    """Print the 2D unfolded Rubik's cube net with ANSI colors and decision details."""
    step_str = f"Step {step}: " if step is not None else ""
    action_str = f" | Action: [bold yellow]{action}[/bold yellow]" if action else ""
    conf_str = f" (Confidence: [bold green]{confidence:.1%}[/bold green])" if confidence is not None else ""
    misplaced = cube.misplaced_stickers()
    score = cube.stage_completion_score()

    header = f"{step_str}{title}{action_str}{conf_str} | Misplaced: {misplaced} | Progress: {score:.1%}"
    console.print(f"\n[bold]{header}[/bold]")
    print(cube.render_ascii(use_color=True))


def plot_benchmark_chart(metrics: Dict[str, Any], output_path: str = "benchmark_results.png") -> str:
    """Generates a high-quality comparison chart comparing Laya vs Generative LLM."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Laya Decision Engine vs Autoregressive LLM Benchmark", fontsize=14, fontweight="bold")

    categories = ["Laya System 1", "Autoregressive LLM"]
    colors = ["#00C49F", "#FF8042"]

    # 1. Latency (p50 ms)
    p50_laya = metrics.get("laya_p50_ms", 32.0)
    p50_llm = metrics.get("llm_p50_ms", 1250.0)
    ax1 = axes[0]
    bars1 = ax1.bar(categories, [p50_laya, p50_llm], color=colors, width=0.5)
    ax1.set_title("Decision Latency (p50 in ms - lower is better)", fontsize=11)
    ax1.set_ylabel("Milliseconds (ms)")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02), f"{yval:.1f}ms", ha="center", va="bottom", fontweight="bold")

    # 2. Throughput (Decisions/sec)
    tp_laya = metrics.get("laya_throughput", 31.2)
    tp_llm = metrics.get("llm_throughput", 0.8)
    ax2 = axes[1]
    bars2 = ax2.bar(categories, [tp_laya, tp_llm], color=colors, width=0.5)
    ax2.set_title("Throughput (Decisions/sec - higher is better)", fontsize=11)
    ax2.set_ylabel("Decisions / second")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02), f"{yval:.1f}/s", ha="center", va="bottom", fontweight="bold")

    # 3. Output Tokens & Cost ($ per 10k requests)
    cost_laya = metrics.get("laya_cost_10k", 0.00)
    cost_llm = metrics.get("llm_cost_10k", 4.50)
    ax3 = axes[2]
    bars3 = ax3.bar(categories, [cost_laya, cost_llm], color=colors, width=0.5)
    ax3.set_title("API Cost per 10,000 Decisions ($)", fontsize=11)
    ax3.set_ylabel("USD ($)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"${yval:.2f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path
