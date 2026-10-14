# 🧠 Laya Decision Engine Labs & Demos

A comprehensive hands-on educational laboratory exploring the **Laya Decision Engine** — an open-source, non-autoregressive **System 1 AI decision model** developed for high-throughput, low-latency structured decisions without token generation overhead.

---

## 📚 Labs Overview

```
laya-demos/
├── lab01_laya_basics/              # 🎯 Lab 01: Core Architecture & Decision Primitives
│   ├── 01_system1_foundations.py   # Non-autoregressive forward passes & question types
│   ├── 02_structured_schemas.py    # Pydantic & JSON Schema direct projection
│   ├── 03_calibrated_confidence.py # RLCD calibrated probabilities & confidence gating
│   └── 04_end_to_end_pipeline.py   # Hybrid guardrail + routing production pipeline
│
├── lab02_llm_vs_laya_benchmark/    # ⚡ Lab 02: Head-to-Head Benchmark: Laya vs LLMs
│   ├── benchmark_comparison.py     # Latency, throughput, cost, and reliability metrics
│   ├── dataset_generator.py        # Curated real-world scenarios dataset
│   └── benchmark_results.png       # Generated high-resolution comparative charts
│
├── lab03_rubik_cube_solver/        # 🎲 Lab 03: Solving the Rubik's Cube with Laya
│   ├── rubik_interactive_demo.py   # Step-by-step solver with ANSI color terminal net
│   ├── rubik_laya_agent.py         # Singmaster move policy & heuristic search agent
│   └── rubik_benchmark.py          # Search velocity & node evaluation benchmark
│
├── common/                         # 🛠️ Shared Utilities
│   ├── cube_engine.py              # 3x3 Rubik's cube simulator & Singmaster engine
│   ├── mock_engine.py              # Live / Mock adapters for Laya and LLM models
│   └── visualizer.py               # Rich terminal tables, ANSI nets & plotters
└── tests/                          # 🧪 Unit Test Suite
```

---

## 🚀 Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/sunnyvale-academy/laya-demos.git
cd laya-demos

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔬 Lab Breakdown

### [Lab 01: How Laya Works (Basics)](file:///Users/denismaggiorotto/Documents/Progetti/Sunnyvale/repos/laya-demos/lab01_laya_basics/README.md)
Explore the core foundations of System 1 non-autoregressive decision models:
* **Zero Token Generation Overhead**: Predictions execute in a single bidirectional encoder pass (~33ms) directly through specialized classification and ordinal heads.
* **Three Typed Question Primitives**:
  * `choice`: Categorical multi-class routing
  * `noul`: Binary boolean decisions (`true` / `false`)
  * `score`: Calibrated ordinal ratings
* **Pydantic Integration**: Maps JSON Schema properties directly to output layers, guaranteeing 100% schema conformance.
* **Calibrated Probabilities**: Uses RLCD (Reinforcement Learning with Calibrated Decisions) against strictly proper scoring rules.

```bash
python3 lab01_laya_basics/01_system1_foundations.py
python3 lab01_laya_basics/02_structured_schemas.py
python3 lab01_laya_basics/03_calibrated_confidence.py
python3 lab01_laya_basics/04_end_to_end_pipeline.py
```

---

### [Lab 02: Laya vs Autoregressive LLM Benchmark](file:///Users/denismaggiorotto/Documents/Progetti/Sunnyvale/repos/laya-demos/lab02_llm_vs_laya_benchmark/README.md)
Direct head-to-head empirical comparison across identical decision and routing tasks:

| Metric / Dimension | Laya Decision Engine | Autoregressive LLM (e.g. GPT-4o-mini) | Advantage |
| :--- | :--- | :--- | :--- |
| **Architecture** | Non-Autoregressive (System 1) | Autoregressive Generative (System 2) | Laya (Single forward pass) |
| **p50 Latency** | **~33 ms** | **~1,250 ms** | 🚀 **~35x Faster** |
| **Throughput** | **~30 req/s** | **~0.8 req/s** | 🚀 **~35x Higher** |
| **Output Token Cost** | **0 tokens** | **~85 tokens / req** | 💰 **100% Token Cost Elimination** |
| **Schema Conformance** | **100% Deterministic** | **90–95% (Occasional formatting flaws)** | Laya (Head-level guarantee) |
| **Confidence Calibration** | **Proper Scoring Rules (RLCD)** | **Uncalibrated Logprobs** | Laya (Empirically Calibrated) |

```bash
python3 lab02_llm_vs_laya_benchmark/benchmark_comparison.py
```

---

### [Lab 03: Solving the Rubik's Cube with Laya](file:///Users/denismaggiorotto/Documents/Progetti/Sunnyvale/repos/laya-demos/lab03_rubik_cube_solver/README.md)
Demonstrates Laya in an interactive, sequential reinforcement learning / control environment:
* **State Representation**: 54 facelet stickers across 6 faces ($U, D, F, B, L, R$).
* **Action Space**: Singmaster notation moves ($U, U', U2, D, D', D2, F, F', F2, B, B', B2, L, L', L2, R, R', R2$).
* **Heuristic Search**: Laya's sub-35ms evaluation enables real-time tree expansion (Best-First Search) that would take 10+ seconds per move with traditional generative LLMs.
* **Terminal Visualizer**: Real-time 2D unfolded cross layout with ANSI color blocks.

```bash
# Run interactive step-by-step solver
python3 lab03_rubik_cube_solver/rubik_interactive_demo.py 3

# Run search velocity benchmark
python3 lab03_rubik_cube_solver/rubik_benchmark.py
```

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest
```
