# Lab 01: Laya Decision Engine Basics

Welcome to Lab 01! This lab teaches you the core principles, architecture, and developer workflow of the **Laya Decision Engine**.

---

## 🎯 Learning Objectives

1. Understand the difference between **System 1 (Non-Autoregressive Decision Engines)** and **System 2 (Autoregressive Generative LLMs)**.
2. Master Laya's three core typed question primitives:
   - `choice`: Multi-class categorical selection
   - `noul`: Binary / boolean decision (`true` / `false`)
   - `score`: Calibrated ordinal score / rating
3. Integrate typed **JSON Schemas** and **Pydantic** models directly with `laya.decide()`.
4. Work with **Calibrated Confidence** (RLCD) and implement **Confidence Gating** for mission-critical pipelines.
5. Build an end-to-end production hybrid architecture (Fast Guardrails + Smart Router + On-Demand System 2 LLM).

---

## 🧪 Lab Scripts

| Script | Topic | Key Takeaway |
| :--- | :--- | :--- |
| [`01_system1_foundations.py`](./01_system1_foundations.py) | Non-Autoregressive Primitives | Learn how `choice`, `noul`, and `score` execute in a single forward pass (~33ms). |
| [`02_structured_schemas.py`](./02_structured_schemas.py) | Pydantic & JSON Schema | Project schemas directly to decision heads with 100% deterministic schema conformance. |
| [`03_calibrated_confidence.py`](./03_calibrated_confidence.py) | Calibrated Probabilities & Gating | Use strictly proper scoring rule confidence for automated routing and human escalation. |
| [`04_end_to_end_pipeline.py`](./04_end_to_end_pipeline.py) | Hybrid Production Pipeline | Combine sub-35ms guardrails and routers to offload 90% of requests from slow LLMs. |

---

## 🚀 Running the Lab

Run all scripts from the repository root:

```bash
python3 lab01_laya_basics/01_system1_foundations.py
python3 lab01_laya_basics/02_structured_schemas.py
python3 lab01_laya_basics/03_calibrated_confidence.py
python3 lab01_laya_basics/04_end_to_end_pipeline.py
```
