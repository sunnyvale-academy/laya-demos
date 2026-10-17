# Lab 04: Hardware Benchmarking — Apple Silicon CPU vs GPU (Metal / MPS)

Welcome to Lab 04! This lab conducts deep hardware performance profiling of the **Laya Decision Engine** comparing **Apple Silicon CPU execution** against **Apple Silicon GPU acceleration via Metal Performance Shaders (MPS)** on Apple Silicon hardware (e.g. **Apple M4 Max**).

---

## 🏗️ Hardware Architecture Context (Apple M4 Max & Unified Memory)

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Apple M4 Max Unified Memory Pool (UMA)               │
│                        (Up to 546 GB/s Bandwidth)                      │
└───────────────────▲────────────────────────────────▲───────────────────┘
                    │ Zero PCIe Copy Latency         │ Zero PCIe Copy Latency
┌───────────────────▼──────────────────┐ ┌───────────▼──────────────────┐
│   14-16 Core High-Performance CPU    │ │     32-40 Core Metal GPU     │
│  (Ultra-low kernel dispatch latency) │ │   (High-throughput MPS FP16)  │
└──────────────────────────────────────┘ └──────────────────────────────┘
```

### Key Hardware Characteristics:
1. **Unified Memory Architecture (UMA)**: The CPU and GPU share the same ultra-high bandwidth unified physical memory. Unlike x86 machines with discrete PCIe GPUs, there is **0ms host-to-device memory copy overhead**.
2. **Apple Silicon GPU via MPS (`torch.device("mps")`)**: Accelerates bidirectional ModernBERT transformer encoders using FP16 tensor cores.
3. **Apple CPU (`torch.device("cpu")`)**: Delivers consistent, highly optimized single-thread and multi-core inference.

---

## 📊 Benchmark Suite Scripts

### 1. `01_single_request_latency.py`
Measures single-request decision latency (p50, p95, p99, mean, std) across CPU and MPS:

```bash
python3 lab04_hardware_benchmarking/01_single_request_latency.py
```

### 2. `02_batch_throughput.py`
Evaluates throughput scaling across batch sizes ($1, 4, 8, 16, 32, 64$):

```bash
python3 lab04_hardware_benchmarking/02_batch_throughput.py
```

### 3. `03_question_scaling.py`
Evaluates marginal latency scaling when adding multiple parallel question heads ($1, 2, 4, 8$ heads):

```bash
python3 lab04_hardware_benchmarking/03_question_scaling.py
```

### 4. `benchmark_cpu_vs_gpu.py` (Full Suite & Plotter)
Runs all benchmarks and generates [`hardware_benchmark_results.png`](file:///Users/denismaggiorotto/Documents/Progetti/Sunnyvale/repos/laya-demos/lab04_hardware_benchmarking/hardware_benchmark_results.png):

```bash
python3 lab04_hardware_benchmarking/benchmark_cpu_vs_gpu.py
```

---

## 📈 Empirical Results Summary (Apple M4 Max)

| Metric / Dimension | Apple Silicon CPU | Apple Silicon GPU (MPS) | MPS Advantage |
| :--- | :--- | :--- | :--- |
| **p50 Latency** | **~223.1 ms** | **~74.0 ms** | 🚀 **~3.0x Faster** |
| **p95 Latency** | **~254.1 ms** | **~79.9 ms** | 🚀 **~3.2x Faster** |
| **Throughput (Single-stream)**| **~4.5 req/s** | **~13.4 req/s** | 🚀 **~3.0x Higher** |
| **8-Head Parallel Latency** | **~227.0 ms** | **~72.8 ms** | 🚀 **~3.1x Faster** |
| **Memory Bus Transfer Tax** | **0 ms (UMA)** | **0 ms (UMA)** | **Identical Shared RAM** |
