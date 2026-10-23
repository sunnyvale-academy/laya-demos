"""Generates labeled trajectory datasets from Snake expert simulations for supervised fine-tuning."""

from __future__ import annotations
import json
import os
import random
from typing import Dict, List, Any, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import print_header, SnakeGame, Direction, get_snake_decision_questions


def generate_snake_dataset(num_games: int = 150, max_steps_per_game: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Plays automated games using the heuristic expert to collect training samples."""
    random.seed(seed)
    dataset: List[Dict[str, Any]] = []

    print_header(
        "Lab 05.1: Generating Nokia 3310 Snake Fine-Tuning Dataset",
        f"Simulating {num_games} Expert Trajectories to Collect Labeled Decision Tuples"
    )

    questions_schema = get_snake_decision_questions()
    total_apples = 0
    total_steps = 0

    for game_idx in range(num_games):
        game = SnakeGame(width=10, height=10, seed=seed + game_idx)
        state_dict = game.get_state_dict()

        for step in range(max_steps_per_game):
            # Expert policy computes the optimal action
            best_action = game.get_expert_action()
            danger = state_dict["danger"]

            # Ground truth targets for all 3 heads
            target_action = best_action.value
            target_danger_ahead = danger["danger_straight"]

            # Path safety score (0: trapped, 1: tight, 2: open, 3: optimal clear)
            if danger["danger_straight"] and danger["danger_left"] and danger["danger_right"]:
                target_safety = 0
            elif danger["danger_straight"]:
                target_safety = 1
            elif not danger["danger_straight"] and (danger["danger_left"] or danger["danger_right"]):
                target_safety = 2
            else:
                target_safety = 3

            sample = {
                "id": f"game_{game_idx:03d}_step_{step:03d}",
                "state_text": state_dict["state_text"],
                "labels": {
                    "action": target_action,
                    "is_danger_ahead": target_danger_ahead,
                    "path_safety_score": target_safety,
                },
                "metadata": {
                    "score": game.score,
                    "snake_length": len(game.snake),
                }
            }
            dataset.append(sample)

            next_state, reward, done = game.step(best_action)
            state_dict = next_state
            total_steps += 1

            if done:
                total_apples += game.score
                break

    print(f"\n[Dataset Generation Complete]:")
    print(f"  • Total Labeled Decisions Collected: {len(dataset):,} samples")
    print(f"  • Total Apples Eaten: {total_apples} apples across {num_games} games")
    print(f"  • Average Snake Length Achieved: {3 + (total_apples / max(1, num_games)):.1f}")

    return dataset


def save_dataset_jsonl(dataset: List[Dict[str, Any]], filepath: str = os.path.join(os.path.dirname(__file__), "snake_dataset.jsonl")) -> str:
    """Saves the dataset in JSONL format."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"  • Saved dataset to: {os.path.abspath(filepath)}")
    return filepath


if __name__ == "__main__":
    data = generate_snake_dataset(num_games=100)
    save_dataset_jsonl(data)
