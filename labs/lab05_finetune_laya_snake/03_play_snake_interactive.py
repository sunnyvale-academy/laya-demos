"""Interactive gameplay demonstration of fine-tuned Laya playing Nokia 3310 Snake in real-time."""

from __future__ import annotations
import os
import sys
import time
from typing import Dict, Any, Optional
import torch

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import print_header, SnakeGame, Direction, get_snake_decision_questions, get_hardware_profile
try:
    from labs.lab05_finetune_laya_snake.train_snake import SnakeFineTunedPolicy, extract_features_from_sample, REV_ACTION_MAP
except ImportError:
    from train_snake import SnakeFineTunedPolicy, extract_features_from_sample, REV_ACTION_MAP


def play_snake_with_finetuned_laya(
    checkpoint_path: str = os.path.join(os.path.dirname(__file__), "snake_checkpoint.pt"),
    max_steps: int = 120,
    fps: float = 8.0,
    seed: Optional[int] = 123
):
    hw = get_hardware_profile()
    device_str = "mps" if hw["mps_available"] else ("cuda" if hw["cuda_available"] else "cpu")
    device = torch.device(device_str)

    print_header(
        "Lab 05.3: Real-Time Nokia 3310 Snake Gameplay with Fine-Tuned Laya",
        f"Inference Running on {hw['cpu_brand']} ({device_str.upper()}) | Model Checkpoint: snake_checkpoint.pt"
    )

    if not os.path.exists(checkpoint_path):
        print(f"[Error: Checkpoint {checkpoint_path} not found. Running training first...]")
        try:
            from labs.lab05_finetune_laya_snake.train_snake import train_snake_model
        except ImportError:
            from train_snake import train_snake_model
        train_snake_model(save_path=checkpoint_path)

    # Load fine-tuned model
    model = SnakeFineTunedPolicy().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    game = SnakeGame(width=10, height=10, seed=seed)
    step = 0
    total_reward = 0.0

    print("\n🎮 [Starting Live Nokia 3310 Snake Game Simulation]:\n")

    while not game.game_over and step < max_steps:
        state_dict = game.get_state_dict()
        sample = {
            "state_text": state_dict["state_text"],
            "metadata": {"snake_length": len(game.snake), "score": game.score}
        }
        feats = torch.tensor([extract_features_from_sample(sample)], dtype=torch.float32).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            act_logits, dan_logit, saf_logits = model(feats)
            probs_act = torch.softmax(act_logits, dim=-1)[0]
            pred_act_idx = torch.argmax(probs_act).item()
            confidence_act = probs_act[pred_act_idx].item()

            p_danger = torch.sigmoid(dan_logit).item()
            safety_lvl = torch.argmax(torch.softmax(saf_logits, dim=-1)).item()
        
        latency_ms = (time.perf_counter() - t0) * 1000.0

        chosen_action = Direction(REV_ACTION_MAP[pred_act_idx])
        safety_labels = ["TRAPPED", "TIGHT", "OPEN", "OPTIMAL"]
        saf_str = safety_labels[safety_lvl] if 0 <= safety_lvl < len(safety_labels) else str(safety_lvl)

        # Clear line / re-render
        print("\033[2J\033[H", end="")
        print(f"╭─────────────────────────────────────────────────────────────╮")
        print(f"│  🕹️  NOKIA 3310 RETRO SNAKE - LAYA DECISION ENGINE AGENT   │")
        print(f"╰─────────────────────────────────────────────────────────────╯")
        print(game.render_nokia(use_color=True))
        print(f" • Step: {step:03d} | Action: [bold cyan]{chosen_action.value}[/bold cyan] ({confidence_act:.1%}) | Safety: [bold green]{saf_str}[/bold green] | Hazard Ahead: {p_danger:.1%} | Latency: {latency_ms:.2f}ms")

        next_state, reward, done = game.step(chosen_action)
        total_reward += reward
        step += 1

        time.sleep(1.0 / fps)

    print("\n" + "=" * 60)
    if game.food == (-1, -1):
        print(f"🏆 [VICTORY!] Snake filled the entire board! Final Score: {game.score}")
    else:
        print(f"🏁 [GAME OVER] Final Apples Eaten: {game.score} | Total Moves: {game.steps_alive} | Total Reward: {total_reward:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    fps_val = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    play_snake_with_finetuned_laya(fps=fps_val, max_steps=80)
