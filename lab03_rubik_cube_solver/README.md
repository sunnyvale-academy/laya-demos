# Lab 03: Solving the Rubik's Cube with Laya Decision Engine

Welcome to Lab 03! This lab explores how the **Laya Decision Engine** operates as a fast **System 1 heuristic policy** in an interactive, sequential control environment: solving a 3x3x3 Rubik's Cube.

---

## 🎲 Concept & Modeling

1. **State Representation**:
   - 54 facelet stickers across 6 faces ($U, D, F, B, L, R$).
   - Formatted into structured textual state descriptors for Laya forward passes.
2. **Action Space**:
   - Singmaster notation quarter-turns: `U`, `U'`, `D`, `D'`, `F`, `F'`, `B`, `B'`, `L`, `L'`, `R`, `R'`.
3. **Laya's Role**:
   - Evaluates state and candidate actions in single forward passes (~30ms).
   - Guides tree search (Best-First Search / Beam Search) to find optimal move sequences without the 1000ms+ token latency bottlenecks of traditional generative LLMs.

---

## 🚀 Running the Lab

### 1. Interactive Step-by-Step Solver

Watch the cube scramble and solve with full ANSI color terminal net rendering:

```bash
# Solve with a 5-step scramble
python3 lab03_rubik_cube_solver/rubik_interactive_demo.py 5
```

### 2. Search Speed Benchmark

Compare Laya's node evaluation speed against theoretical autoregressive LLM lookahead:

```bash
python3 lab03_rubik_cube_solver/rubik_benchmark.py
```
