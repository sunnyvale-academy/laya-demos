# Hands-on Exercise 01: Teaching AI to Make Decisions with LAYA (Spam Test)

**Duration:** 20–30 minutes  
**Level:** Beginner  
**Tools:** Python and LAYA Decision Engine

---

## 🎯 Objective

Understand how a **System 1 decision model** makes fast, deterministic, structured decisions based on incoming data, and how it seamlessly pairs with **System 2 reasoning models** to handle ambiguous edge cases.

---

## 🛡️ The Challenge: Should We Block This Email?

Imagine you are building an automated email security gateway for an enterprise. For every incoming email, the AI must decide whether to **`ALLOW`** or **`BLOCK`** it.

### Step 1 — Define the Scenarios

We prepare five representative email scenarios exhibiting varying security characteristics:

| Email | Sender | Suspicious Link | Urgent Language | Expected Decision |
| :--- | :--- | :---: | :---: | :--- |
| **A** | Known colleague | No | No | `ALLOW` |
| **B** | Unknown sender | Yes | Yes | `BLOCK` |
| **C** | Known bank | No | No | `ALLOW` |
| **D** | Unknown sender | Yes | Yes | `BLOCK` |
| **E** | Unknown sender | No | Yes | *Discuss / Ambiguous* |

---

### Step 2 — Ask LAYA to Decide

We formulate a typed binary decision using LAYA's supported typed-decision interface:

```python
questions = {
    "security_decision": {
        "type": "choice",
        "instructions": "Decide whether to ALLOW or BLOCK this incoming email.",
        "criteria": {
            "ALLOW": "trusted sender, known colleague, verified bank, clean legitimate communication",
            "BLOCK": "unknown sender, suspicious link, phishing threat, malicious url, credential harvesting, scam",
        },
    }
}
```

For each email scenario, we feed the contextual state into LAYA:
- LAYA executes in a single forward pass (**~30ms**)
- It outputs a typed decision (`ALLOW` or `BLOCK`) along with calibrated probability / confidence (`confidence`).

---

### Step 3 — Evaluate the Results

Compare LAYA's decisions with expected security policies:

| Email | Sender | Link? | Urgent? | LAYA (Sys 1) | Confidence | Latency | Policy Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** | Known colleague | No | No | **ALLOW** | 95.4% | ~35 ms | ✅ MATCH |
| **B** | Unknown sender | Yes | Yes | **BLOCK** | 97.8% | ~35 ms | ✅ MATCH |
| **C** | Known bank | No | No | **ALLOW** | 98.2% | ~35 ms | ✅ MATCH |
| **D** | Unknown sender | Yes | Yes | **BLOCK** | 95.3% | ~35 ms | ✅ MATCH |
| **E** | Unknown sender | No | Yes | **BLOCK** | 78.0% | ~35 ms | ⚖️ Edge Case |

#### Key Discussion Points
1. **How many decisions were correct?**
   Emails A, B, C, and D were classified with >95% confidence, matching security expectations.
2. **Which emails were difficult to classify?**
   Email E is inherently ambiguous: an unknown vendor is requesting an RFP meeting with urgent timing, but contains *no* phishing links or malware payloads.
3. **How confident was the model when it faced ambiguity?**
   LAYA flagged Email E with significantly lower confidence (~78%), providing a clean boundary for confidence-gated escalation.
4. **Would you trust this model to block emails automatically?**
   For high-confidence decisions (>85%), yes. For lower-confidence edge cases, pure automated blocking risks severe false positives (e.g. missing critical business RFPs).

---

### Step 4 — Introduce System 2 (Reasoning Model)

For ambiguous Email E, we escalate to a System 2 reasoning model to perform deep evidence analysis and Chain-of-Thought (CoT) audit:

```
[System 2 Audit]
1. Evidence Analysis:
   - Sender is unknown to the organization.
   - No suspicious links or malicious attachments are detected.
   - Urgent language corresponds to a legitimate business context (an RFP deadline).
2. Risk Assessment:
   - Probability of Malicious Payload: LOW (< 5%)
   - Risk of False Positive Disruption: HIGH (missed contract opportunity)
3. Recommendation:
   - ALLOW with tag: [EXTERNAL SENDER - PROCEED WITH CAUTION]
```

### System 1 vs System 2 Comparison

| Dimension | System 1 (LAYA) | System 2 (Reasoning Model) | Optimal Production Strategy |
| :--- | :--- | :--- | :--- |
| **Role** | Instant Triage & Binary Filter | Deep Evidence Reasoner & Auditor | Use LAYA for 95% of routine emails |
| **Latency** | **~30 ms** (Sub-second) | **~1,200 ms** (Multi-second) | Escalate only low-confidence cases (<85%) |
| **Token Cost** | **$0.00** (Zero tokens generated) | ~85 tokens / decision | Reduces LLM API billing by 90%+ |
| **Output Format** | Strict Typed Choice (`ALLOW` / `BLOCK`) | Free-form Explanation & Steps | Combines speed with full explainability |

---

## 🎯 Bonus Experiment: Sensitivity Analysis

What happens if we modify a single characteristic of Email E?

| Scenario | Sender | Link? | Urgent? | LAYA Decision | Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Original Email E** | Unknown sender | No | Yes | `BLOCK` (Borderline) | 78.0% |
| **Variation 1: Add Suspicious Link** | Unknown sender | **Yes** | Yes | `BLOCK` (High risk) | 96.1% |
| **Variation 2: Trusted Sender** | **Known colleague** | No | Yes | `ALLOW` (Trusted) | 98.3% |

Changing even one attribute shifts the decision boundaries cleanly, demonstrating LAYA's responsiveness to multi-feature input states.

---

## 🚀 Running the Exercise

Run the exercise directly from the repository root:

```bash
python3 exercises/exercise01_spam_test/spam_detector.py
```

Or using the project's virtual environment:

```bash
PYTHONPATH=.venv/lib/python3.11/site-packages:. python3 exercises/exercise01_spam_test/spam_detector.py
```

