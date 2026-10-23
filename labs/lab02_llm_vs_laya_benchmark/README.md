# Lab 02: Head-to-Head Benchmark: Laya vs Autoregressive LLMs

Welcome to Lab 02! This lab presents an empirical, head-to-head comparison between the **Laya Decision Engine** (System 1) and standard **Autoregressive Large Language Models** (System 2, e.g., GPT-4o-mini, Claude, Llama).

---

## 📊 Benchmark Metrics Tracked

1. **Latency Distribution**:
   - p50, p95, and p99 latency in milliseconds.
   - Demonstrates the ~30–40x speedup from eliminating autoregressive token generation.
2. **Throughput**:
   - Single-threaded decisions per second.
3. **Token Usage & API Cost**:
   - 0 output tokens vs 60–100 output tokens per decision.
   - Cost scaling per 10,000 decisions.
4. **Schema Reliability & Formatting Flaws**:
   - Deterministic 100% schema compliance at the neural network head level vs markdown/JSON parsing errors in LLMs.
5. **Calibrated Confidence**:
   - Mathematically calibrated proper scoring rule probabilities vs uncalibrated LLM generation.

---

## 🚀 Running the Benchmark

```bash
python3 labs/lab02_llm_vs_laya_benchmark/benchmark_comparison.py
```

This will run the full test suite, print a structured comparison table in the console, and generate a visualization chart at `labs/lab02_llm_vs_laya_benchmark/benchmark_results.png`.
