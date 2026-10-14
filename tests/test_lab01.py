import pytest
from common import create_laya_agent
import laya.presets as presets


def test_laya_agent_predict():
    agent = create_laya_agent()
    questions = presets.triage_questions()
    res = agent.predict(
        state={"message": "I was double charged, refund me immediately."},
        questions=questions
    )
    assert "answers" in res
    assert "confidences" in res
    assert res["answers"]["intent"] in ["refund", "billing_question", "technical_help", "information", "cancellation", "other"]
    assert res["elapsed_ms"] > 0


def test_laya_agent_decide_schema():
    agent = create_laya_agent()
    schema = {
        "type": "object",
        "properties": {
            "department": {"type": "string", "enum": ["billing", "support", "sales"]},
            "urgent": {"type": "boolean"},
            "level": {"type": "integer", "maximum": 3},
        }
    }
    answers = agent.decide("Help my server crashed urgently!", schema=schema)
    assert "department" in answers
    assert "urgent" in answers
    assert answers["urgent"] is True
