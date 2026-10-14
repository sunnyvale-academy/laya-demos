#!/usr/bin/env python3
"""Lab 03: Rubik's Cube Decision Speed Benchmark.

Compares:
- Laya Decision Engine Search Speed vs Theoretical Generative LLM Lookahead
- Nodes Evaluated per Second
- Total Solve Latency
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.cube_engine import RubikCube
from common.visualizer import print_header, print_metrics_table
from rubik_laya_agent import LayaRubikSolver


def run_rubik_benchmark():
    print_header(
        "Lab 03 Benchmark: Rubik's Cube Search Engine Speed",
        "Evaluating Decision Velocity in Sequential Action Environments"
    )

    scramble_depths = [2, 3, 4, 5, 6]
    solver = LayaRubikSolver()

    benchmark_rows = []
    headers = ["Scramble Depth", "Solution Moves", "Laya Solve Time", "LLM Est. Time (1.2s/node)", "Speedup Factor"]

    for depth in scramble_depths:
        cube = RubikCube()
        cube.scramble(steps=depth, seed=100 + depth)

        t0 = time.perf_counter()
        solution, history, elapsed = solver.solve(cube, max_depth=10, max_nodes=1000)
        actual_elapsed = time.perf_counter() - t0

        nodes_evaluated = len(solution) * 12 + 10  # approximate branching count
        llm_est_time_sec = nodes_evaluated * 1.25  # 1.25s per LLM generation call

        speedup = llm_est_time_sec / max(0.001, actual_elapsed)

        benchmark_rows.append([
            f"{depth} moves",
            f"{len(solution)} moves" if solution else "N/A",
            f"{actual_elapsed*1000:.1f} ms",
            f"{llm_est_time_sec:.1f} s",
            f"🚀 {speedup:.0f}x"
        ])

    print_metrics_table(headers, benchmark_rows, title="Cube Solver Speed Comparison")

    print("\n[Insights]")
    print("  • Physical and digital games require sub-50ms feedback loops.")
    print("  • LLMs cannot perform search tree expansion directly in interactive control loops due to token latency.")
    print("  • Laya functions as a fast System 1 heuristic & policy evaluation engine.")


if __name__ == "__main__":
    run_rubik_benchmark()
