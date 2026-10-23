import pytest
import os
try:
    import torch
except (ImportError, Exception):
    torch = None

if torch is None:
    pytest.skip("PyTorch not installed or incompatible C extensions", allow_module_level=True)

from common import SnakeGame, Direction, get_snake_decision_questions
try:
    from labs.lab05_finetune_laya_snake.train_snake import (
        SnakeFineTunedPolicy,
        extract_features_from_sample,
        ACTION_MAP,
        REV_ACTION_MAP,
    )
except ImportError:
    from lab05_finetune_laya_snake.train_snake import (
        SnakeFineTunedPolicy,
        extract_features_from_sample,
        ACTION_MAP,
        REV_ACTION_MAP,
    )


def test_snake_game_environment():
    game = SnakeGame(width=10, height=10, seed=42)
    state = game.get_state_dict()
    assert "state_text" in state
    assert "head" in state
    assert "food" in state
    assert game.score == 0
    assert len(game.snake) == 3

    # Step in valid direction
    next_state, reward, done = game.step(Direction.RIGHT)
    assert not done
    assert game.steps_alive == 1


def test_snake_decision_questions_schema():
    questions = get_snake_decision_questions()
    assert "action" in questions
    assert questions["action"]["type"] == "choice"
    assert "is_danger_ahead" in questions
    assert questions["is_danger_ahead"]["type"] == "noul"
    assert "path_safety_score" in questions
    assert questions["path_safety_score"]["type"] == "score"


def test_snake_feature_extraction():
    game = SnakeGame(width=10, height=10, seed=42)
    state_dict = game.get_state_dict()
    sample = {
        "state_text": state_dict["state_text"],
        "metadata": {"snake_length": 3, "score": 0}
    }
    feats = extract_features_from_sample(sample)
    assert len(feats) == 16
    assert all(isinstance(f, float) for f in feats)


def test_finetuned_policy_forward():
    model = SnakeFineTunedPolicy(hidden_dim=64)
    x = torch.randn(2, 16)
    act_logits, dan_logit, saf_logits = model(x)
    assert act_logits.shape == (2, 4)
    assert dan_logit.shape == (2,)
    assert saf_logits.shape == (2, 4)
