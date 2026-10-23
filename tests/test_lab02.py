import pytest
from common import MockLLMRunner, create_laya_agent
try:
    from labs.lab02_llm_vs_laya_benchmark.dataset_generator import get_benchmark_dataset
except ImportError:
    from lab02_llm_vs_laya_benchmark.dataset_generator import get_benchmark_dataset
import laya.presets as presets


def test_benchmark_dataset():
    data = get_benchmark_dataset()
    assert len(data) >= 5
    for item in data:
        assert "text" in item
        assert "id" in item


def test_llm_runner_decide():
    runner = MockLLMRunner(latency_ms=10.0)
    res = runner.decide("I need a refund for my order", questions=presets.triage_questions())
    assert "answers" in res
    assert "cost_usd" in res
    assert res["is_schema_valid"] is True
