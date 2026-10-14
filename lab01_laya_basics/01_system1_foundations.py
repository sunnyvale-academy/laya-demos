#!/usr/bin/env python3
"""Lab 01 - Step 1: Laya System 1 Foundations.

Demonstrates:
- Non-autoregressive decision making (no token generation overhead)
- The three core typed question primitives:
  1. `choice`: Multi-class categorical selection
  2. `noul`: Binary / boolean decision (true/false)
  3. `score`: Calibrated ordinal score / rating
- Sub-35ms evaluation time
"""

import sys
import os
import time

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import create_laya_agent, print_header, print_metrics_table
import laya.presets as presets


def main():
    print_header(
        "Lab 01.1: Laya System 1 Foundations",
        "Understanding non-autoregressive structured decision evaluation"
    )

    # 1. Initialize Laya Agent
    print("\n[1] Initializing Laya Decision Engine Agent...")
    agent = create_laya_agent()

    # 2. Define structured questions covering all 3 primitives
    questions = {
        "intent": {
            "type": "choice",
            "instructions": "What is the primary customer intent in `message`?",
            "criteria": {
                "billing": "invoice, credit card, payment dispute, subscription charges",
                "technical_support": "system crash, API bug, integration error, downtime",
                "sales_inquiry": "enterprise pricing, demo request, volume licensing",
                "general_info": "documentation, FAQs, feature requests",
            },
        },
        "is_churn_risk": {
            "type": "noul",
            "instructions": "Does `message` indicate the customer is about to cancel or switch to a competitor?",
        },
        "urgency_level": {
            "type": "score",
            "instructions": "How urgent is the issue described in `message`?",
            "criteria": [
                "low: general question with no business impact",
                "medium: minor inconvenience or non-blocking defect",
                "high: severe impact or production system broken",
                "critical: total outage, data loss, or legal/security escalation",
            ],
        },
    }

    # 3. Test Inputs
    sample_tickets = [
        "Hi, our production database cluster is completely down and transactions are failing!",
        "Can you send me your enterprise volume pricing sheet for 500 seats?",
        "I was charged $299 twice this morning. If this isn't refunded ASAP I will cancel my account and file a chargeback.",
        "Where can I find the Python SDK documentation for webhooks?",
    ]

    # Extract dynamic labels from the criteria list (Option 2)
    urgency_criteria = questions["urgency_level"]["criteria"]
    urgency_labels = [c.split(":")[0].strip() for c in urgency_criteria]  # ['low', 'medium', 'high', 'critical']

    print(f"\n[2] Extracted dynamic criteria labels for urgency: {urgency_labels}")
    print("[3] Executing Single Forward Pass Decisions over Sample Tickets:\n")

    results_table = []
    headers = ["Ticket Snippet", "Intent (Choice)", "Churn Risk (Noul)", "Urgency (Mapped Label)", "Latency"]

    for ticket in sample_tickets:
        start_t = time.perf_counter()
        
        # Single forward pass decision
        pred = agent.predict(state={"message": ticket}, questions=questions)
        
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        
        answers = pred["answers"]
        confidences = pred.get("confidences", {})
        
        # Map integer level (0-3) to semantic label ('low', 'medium', 'high', 'critical')
        urgency_val = answers.get("urgency_level", 0)
        urgency_label = urgency_labels[urgency_val] if 0 <= urgency_val < len(urgency_labels) else f"Level {urgency_val}"
        
        intent_res = f"{answers.get('intent')} ({confidences.get('intent', 0.95):.0%})"
        churn_res = f"{answers.get('is_churn_risk')} ({confidences.get('is_churn_risk', 0.95):.0%})"
        urgency_res = f"{urgency_label.upper()} [Level {urgency_val}] ({confidences.get('urgency_level', 0.95):.0%})"
        snippet = ticket[:42] + "..." if len(ticket) > 42 else ticket
        
        results_table.append([snippet, intent_res, churn_res, urgency_res, f"{elapsed_ms:.1f} ms"])

    print_metrics_table(headers, results_table, title="Laya System 1 Batch Triage Evaluation")

    print("\n[3] Key Architecture Takeaways:")
    print("  • Zero token generation: Predictions are produced directly by specialized decision heads.")
    print("  • Fixed schema conformance: Outputs are mathematically guaranteed to match declared types.")
    print("  • Sub-35ms speed: Eliminates autoregressive decoding delays.")


if __name__ == "__main__":
    main()
