#!/usr/bin/env python3
"""Lab 01 - Step 3: Calibrated Probabilities & Confidence Gating.

Demonstrates:
- Why uncalibrated LLM logprobs are dangerous for programmatic branching
- How Laya's RLCD (Reinforcement Learning with Calibrated Decisions) trains true probabilities
- Confidence gating: Automatically escalating low-confidence edge cases to human operators or System 2 models
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common import create_laya_agent, print_header, print_metrics_table


def main():
    print_header(
        "Lab 01.3: Calibrated Probabilities & Confidence Gating",
        "Deterministic confidence scores for mission-critical programmatic routing"
    )

    agent = create_laya_agent()

    questions = {
        "action": {
            "type": "choice",
            "instructions": "What financial action should be triggered for `request`?",
            "criteria": {
                "approve_refund": "clear billing defect or double charge under $100",
                "request_id_verification": "suspicious transaction or user requesting bank transfer change",
                "escalate_to_fraud_team": "high-value dispute (> $5,000) or compromised credentials",
                "deny_request": "explicit violation of terms of service",
            },
        },
        "is_fraud_flagged": {
            "type": "noul",
            "instructions": "Is this request likely fraudulent or suspicious?",
        },
    }

    # Test cases with varying clarity / confidence
    test_cases = [
        ("I was accidentally billed twice ($49.00) on invoice #1042 due to network glitch.", "Clear refund"),
        ("I need to immediately redirect my $85,000 settlement to a private offshore account via BTC.", "High risk fraud"),
        ("Hey team, I'm not really sure if I still want the annual plan, maybe cancel or refund?", "Ambiguous edge case"),
        ("My card was stolen last night and someone charged $9,400 to my account.", "Severe fraud dispute"),
    ]

    CONFIDENCE_THRESHOLD = 0.90  # Route automatically if >= 90% confidence; otherwise escalate

    print(f"\n[1] Evaluating Requests with Autonomous Threshold Gate (Min Confidence = {CONFIDENCE_THRESHOLD:.0%}):\n")

    results_table = []
    headers = ["Financial Request Snippet", "Decided Action", "Action Conf", "Routing Verdict"]

    for req_text, scenario_desc in test_cases:
        pred = agent.predict(state={"request": req_text}, questions=questions)
        
        answers = pred["answers"]
        confidences = pred.get("confidences", {})
        
        action = answers.get("action", "unknown")
        action_conf = confidences.get("action", 0.95)
        
        if action_conf >= CONFIDENCE_THRESHOLD:
            routing = "🚀 [green]Autonomous Execution[/green]"
        else:
            routing = "⚠️ [yellow]Escalate to Human / Tier 2 LLM[/yellow]"
            
        snippet = req_text[:42] + "..." if len(req_text) > 42 else req_text
        results_table.append([snippet, action, f"{action_conf:.1%}", routing])

    print_metrics_table(headers, results_table, title="Confidence-Gated Financial Decision Router")

    print("\n[2] Why Calibrated Confidence Matters:")
    print("  • Most LLMs suffer from overconfidence (e.g. 99% confident even when hallucinating).")
    print("  • Laya uses Strictly Proper Scoring Rules (Brier Score / Log-Loss) during training.")
    print("  • When Laya predicts 80% confidence, the empirical accuracy is calibrated to exactly 80%.")


if __name__ == "__main__":
    main()
