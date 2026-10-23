import pytest
from exercises.exercise01_spam_test.spam_detector import (
    get_email_scenarios,
    get_laya_decision_questions,
    format_state,
    run_system1_evaluation,
    run_system2_analysis,
)
from common import create_laya_agent, MockLLMRunner


def test_email_scenarios_structure():
    scenarios = get_email_scenarios()
    assert len(scenarios) == 5
    ids = [s["id"] for s in scenarios]
    assert ids == ["A", "B", "C", "D", "E"]
    for s in scenarios:
        assert "sender" in s
        assert "suspicious_link" in s
        assert "urgent_language" in s
        assert "expected" in s
        assert "sample_snippet" in s


def test_laya_decision_questions():
    questions = get_laya_decision_questions()
    assert "security_decision" in questions
    assert questions["security_decision"]["type"] == "choice"
    criteria = questions["security_decision"]["criteria"]
    assert "ALLOW" in criteria
    assert "BLOCK" in criteria


def test_system1_evaluation():
    agent = create_laya_agent()
    scenarios = get_email_scenarios()
    results = run_system1_evaluation(agent, scenarios)

    assert len(results) == 5
    for res in results:
        assert res["decision"] in ["ALLOW", "BLOCK"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert res["latency_ms"] >= 0.0

    # Email A (Colleague, clean) -> ALLOW
    assert results[0]["decision"] == "ALLOW"
    assert results[0]["confidence"] > 0.85

    # Email B (Unknown, suspicious link) -> BLOCK
    assert results[1]["decision"] == "BLOCK"
    assert results[1]["confidence"] > 0.85

    # Email C (Bank, clean) -> ALLOW
    assert results[2]["decision"] == "ALLOW"
    assert results[2]["confidence"] > 0.85

    # Email D (Unknown, suspicious link) -> BLOCK
    assert results[3]["decision"] == "BLOCK"
    assert results[3]["confidence"] > 0.85

    # Email E is ambiguous -> confidence < 0.85
    assert results[4]["confidence"] < 0.85


def test_system2_analysis():
    llm = MockLLMRunner()
    scenarios = get_email_scenarios()
    email_e = scenarios[4]

    res = run_system2_analysis(llm, email_e)
    assert "explanation" in res
    assert "Evidence Analysis" in res["explanation"]
    assert res["tokens"] > 0
    assert res["cost_usd"] > 0
    assert res["elapsed_ms"] > 0
