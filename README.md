# 🧠 Laya Decision Engine Labs & Demos

A comprehensive hands-on educational laboratory exploring the **Laya Decision Engine** — an open-source, non-autoregressive **System 1 AI decision model** developed for high-throughput, low-latency structured decisions without token generation overhead.

---

## 📚 Labs Overview

```
laya-demos/
├── labs/                             # 🔬 Laya Labs Collection
│   ├── lab01_laya_basics/            # 🎯 Lab 01: Core Architecture & Decision Primitives
│   │   ├── 01_system1_foundations.py # Non-autoregressive forward passes & question types
│   │   ├── 02_structured_schemas.py  # Pydantic & JSON Schema direct projection
│   │   ├── 03_calibrated_confidence.py# RLCD calibrated probabilities & confidence gating
│   │   └── 04_end_to_end_pipeline.py # Hybrid guardrail + routing production pipeline
│   │
│   ├── lab02_llm_vs_laya_benchmark/  # ⚡ Lab 02: Head-to-Head Benchmark: Laya vs LLMs
│   │   ├── benchmark_comparison.py   # Latency, throughput, cost, and reliability metrics
│   │   ├── dataset_generator.py      # Curated real-world scenarios dataset
│   │   └── benchmark_results.png     # Generated high-resolution comparative charts
│   │
│   ├── lab03_rubik_cube_solver/      # 🎲 Lab 03: Solving the Rubik's Cube with Laya
│   │   ├── rubik_interactive_demo.py # Step-by-step solver with ANSI color terminal net
│   │   ├── rubik_laya_agent.py       # Singmaster move policy & heuristic search agent
│   │   └── rubik_benchmark.py        # Search velocity & node evaluation benchmark
│   │
│   ├── lab04_hardware_benchmarking/  # 💻 Lab 04: Hardware Profiling (CPU vs Apple GPU / MPS)
│   │   ├── 01_single_request_latency.py# Single-request latency profiling on Apple Silicon
│   │   ├── 02_batch_throughput.py    # Batch scaling and saturation benchmark
│   │   ├── 03_question_scaling.py    # Multi-head question complexity scaling
│   │   ├── benchmark_cpu_vs_gpu.py   # Comprehensive benchmark suite & chart plotter
│   │   └── hardware_benchmark_results.png # Generated comparison visualization
│   │
│   └── lab05_finetune_laya_snake/    # 🕹️ Lab 05: Fine-Tuning Laya on Nokia 3310 Snake AI
│       ├── 01_generate_snake_dataset.py# Trajectory generator from heuristic expert
│       ├── 02_train_laya_snake.py    # Multi-task supervised fine-tuning training loop
│       ├── 03_play_snake_interactive.py# Live interactive retro gameplay CLI
│       ├── 04_evaluate_snake.py      # 50-episode benchmark vs Random Policy & A*
│       └── snake_benchmark_results.png# Policy performance & throughput charts
│
├── exercises/                        # 📝 Hands-on Exercises & Practice Challenges
├── common/                           # 🛠️ Shared Utilities
│   ├── cube_engine.py                # 3x3 Rubik's cube simulator & Singmaster engine
│   ├── snake_engine.py               # Nokia 3310 Snake simulator & ANSI visualizer
│   ├── hardware_profile.py           # Apple Silicon & GPU hardware detection
│   ├── mock_engine.py                # Live / Mock adapters for Laya and LLM models
│   └── visualizer.py                 # Rich terminal tables, ANSI nets & plotters
└── tests/                            # 🧪 Unit Test Suite (14/14 Passing)
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

### [Lab 01: How Laya Works (Basics)](labs/lab01_laya_basics/README.md)
Explore the core foundations of System 1 non-autoregressive decision models:
* **Zero Token Generation Overhead**: Predictions execute in a single bidirectional encoder pass directly through specialized classification and ordinal heads.
* **Three Typed Question Primitives**: `choice` (categorical enum), `noul` (boolean), `score` (calibrated ordinal ranking).
* **Pydantic Integration**: Maps JSON Schema properties directly to output layers, guaranteeing 100% schema conformance.
* **Calibrated Probabilities**: Uses RLCD (proper scoring rules) with dynamic criteria extraction.

```bash
python3 labs/lab01_laya_basics/01_system1_foundations.py
python3 labs/lab01_laya_basics/02_structured_schemas.py
python3 labs/lab01_laya_basics/03_calibrated_confidence.py
python3 labs/lab01_laya_basics/04_end_to_end_pipeline.py
```

---

### [Lab 02: Laya vs Autoregressive LLM Benchmark](labs/lab02_llm_vs_laya_benchmark/README.md)
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
python3 labs/lab02_llm_vs_laya_benchmark/benchmark_comparison.py
```

---

### [Lab 03: Solving the Rubik's Cube with Laya](labs/lab03_rubik_cube_solver/README.md)
Demonstrates Laya in an interactive, sequential reinforcement learning / control environment:
* **State Representation**: 54 facelet stickers across 6 faces ($U, D, F, B, L, R$).
* **Action Space**: Singmaster notation moves (`U`, `U'`, `F`, `F'`, `R`, `R'`, `D`, `D'`, `L`, `L'`, `B`, `B'`).
* **Heuristic Search**: Laya's sub-35ms evaluation enables real-time tree expansion (Best-First Search).
* **Terminal Visualizer**: Real-time 2D unfolded cross layout with ANSI color blocks.

```bash
# Run interactive step-by-step solver
python3 labs/lab03_rubik_cube_solver/rubik_interactive_demo.py 3

# Run search velocity benchmark
python3 labs/lab03_rubik_cube_solver/rubik_benchmark.py
```

---

### [Lab 04: Hardware Benchmarking — Apple Silicon CPU vs GPU (MPS)](labs/lab04_hardware_benchmarking/README.md)
Deep profiling on Apple Silicon (e.g. **Apple M4 Max** with Unified Memory Architecture):

| Metric / Dimension | Apple Silicon CPU | Apple Silicon GPU (MPS) | MPS Advantage |
| :--- | :--- | :--- | :--- |
| **Single p50 Latency** | **~223.1 ms** | **~74.0 ms** | 🚀 **~3.0x Faster** |
| **Single p95 Latency** | **~254.1 ms** | **~79.9 ms** | 🚀 **~3.2x Faster** |
| **Single Throughput** | **~4.5 req/s** | **~13.4 req/s** | 🚀 **~3.0x Higher** |
| **8-Head Parallel Latency** | **~205.9 ms** | **~69.0 ms** | 🚀 **~3.0x Faster** |
| **Memory Bus Overhead** | **0 ms (UMA)** | **0 ms (UMA)** | **Shared Unified RAM** |

```bash
# Run individual benchmarks
python3 labs/lab04_hardware_benchmarking/01_single_request_latency.py
python3 labs/lab04_hardware_benchmarking/02_batch_throughput.py
python3 labs/lab04_hardware_benchmarking/03_question_scaling.py

# Run full hardware suite & export chart
python3 labs/lab04_hardware_benchmarking/benchmark_cpu_vs_gpu.py
```

---

### [Lab 05: Fine-Tuning Laya on Nokia 3310 Snake AI](labs/lab05_finetune_laya_snake/README.md)
Supervised fine-tuning of Laya decision heads on a retro Nokia 3310 Snake environment:
* **Environment**: 10x10 Snake grid with boundary collision and dynamic apple spawning.
* **Dataset Generation**: Heuristic expert runs collecting state representations and actions.
* **Multi-Task Heads**: Simultaneous learning for action selection (`choice`), danger detection (`noul`), and path safety scoring (`score`).
* **Interactive Gameplay**: Real-time ANSI game visualization rendered in the terminal.

```bash
# 1. Generate expert trajectory dataset
python3 labs/lab05_finetune_laya_snake/01_generate_snake_dataset.py

# 2. Train fine-tuned policy adapter
python3 labs/lab05_finetune_laya_snake/02_train_laya_snake.py

# 3. Play interactive CLI snake game
python3 labs/lab05_finetune_laya_snake/03_play_snake_interactive.py 15

# 4. Run policy evaluation benchmark
python3 labs/lab05_finetune_laya_snake/04_evaluate_snake.py
```

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest
```
