"""Fine-tuning training loop for Laya Decision Engine on Nokia 3310 Snake policy & safety."""

from __future__ import annotations
import json
import os
import random
import time
from typing import Dict, List, Any, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import print_header, print_metrics_table, get_hardware_profile, get_snake_decision_questions


class SnakeFineTunedPolicy(nn.Module):
    """Fine-tuned multi-task decision adapter built on top of Laya representations."""

    def __init__(self, hidden_dim: int = 256, num_actions: int = 4, num_safety_scores: int = 4):
        super().__init__()
        # State feature encoder
        self.feature_extractor = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Multi-task Laya Decision Heads
        self.action_head = nn.Linear(hidden_dim, num_actions)        # Choice head (UP, DOWN, LEFT, RIGHT)
        self.danger_head = nn.Linear(hidden_dim, 1)                  # Noul head (P(danger_ahead))
        self.safety_head = nn.Linear(hidden_dim, num_safety_scores)  # Score head (0..3 ordinal)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        action_logits = self.action_head(features)
        danger_logit = self.danger_head(features).squeeze(-1)
        safety_logits = self.safety_head(features)
        return action_logits, danger_logit, safety_logits


def extract_features_from_sample(sample: Dict[str, Any]) -> List[float]:
    """Encodes state text / metadata into normalized feature vector."""
    txt = sample["state_text"].lower()
    
    # 16-dimensional spatial descriptor
    f_danger_straight = 1.0 if "obstacle in front: true" in txt else 0.0
    f_danger_right = 1.0 if "obstacle on right: true" in txt else 0.0
    f_danger_left = 1.0 if "obstacle on left: true" in txt else 0.0
    f_north = 1.0 if "north" in txt else 0.0
    f_south = 1.0 if "south" in txt else 0.0
    f_east = 1.0 if "east" in txt else 0.0
    f_west = 1.0 if "west" in txt else 0.0
    f_moving_up = 1.0 if "moving up" in txt else 0.0
    f_moving_down = 1.0 if "moving down" in txt else 0.0
    f_moving_left = 1.0 if "moving left" in txt else 0.0
    f_moving_right = 1.0 if "moving right" in txt else 0.0

    length = sample.get("metadata", {}).get("snake_length", 3) / 100.0
    score = sample.get("metadata", {}).get("score", 0) / 50.0

    # Add interaction features
    f_safe_north = 1.0 if (f_north and not f_danger_straight) else 0.0
    f_safe_south = 1.0 if (f_south and not f_danger_straight) else 0.0
    f_safe_east = 1.0 if (f_east and not f_danger_straight) else 0.0

    return [
        f_danger_straight, f_danger_right, f_danger_left,
        f_north, f_south, f_east, f_west,
        f_moving_up, f_moving_down, f_moving_left, f_moving_right,
        length, score,
        f_safe_north, f_safe_south, f_safe_east,
    ]


ACTION_MAP = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
REV_ACTION_MAP = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}


def train_snake_model(
    dataset_path: str = os.path.join(os.path.dirname(__file__), "snake_dataset.jsonl"),
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_path: str = os.path.join(os.path.dirname(__file__), "snake_checkpoint.pt")
) -> Dict[str, Any]:
    hw = get_hardware_profile()
    device_str = "mps" if hw["mps_available"] else ("cuda" if hw["cuda_available"] else "cpu")
    device = torch.device(device_str)

    print_header(
        "Lab 05.2: Fine-Tuning Laya on Nokia 3310 Snake Policy",
        f"Training Multi-Task Heads on {hw['cpu_brand']} using {device_str.upper()}"
    )

    # 1. Load dataset
    samples: List[Dict[str, Any]] = []
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    random.shuffle(samples)
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n[1] Prepared Dataset: {len(train_samples):,} Training Samples | {len(val_samples):,} Validation Samples")

    # Convert to tensors
    def vectorize_split(data: List[Dict[str, Any]]):
        x = torch.tensor([extract_features_from_sample(s) for s in data], dtype=torch.float32)
        y_act = torch.tensor([ACTION_MAP[s["labels"]["action"]] for s in data], dtype=torch.long)
        y_dan = torch.tensor([1.0 if s["labels"]["is_danger_ahead"] else 0.0 for s in data], dtype=torch.float32)
        y_saf = torch.tensor([s["labels"]["path_safety_score"] for s in data], dtype=torch.long)
        return x, y_act, y_dan, y_saf

    x_train, y_act_tr, y_dan_tr, y_saf_tr = vectorize_split(train_samples)
    x_val, y_act_va, y_dan_va, y_saf_va = vectorize_split(val_samples)

    # Initialize model
    model = SnakeFineTunedPolicy().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Loss functions
    crit_action = nn.CrossEntropyLoss()
    crit_danger = nn.BCEWithLogitsLoss()
    crit_safety = nn.CrossEntropyLoss()

    print(f"[2] Beginning Training across {epochs} Epochs (Batch size: {batch_size})...")

    history: List[Dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(x_train))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(x_train), batch_size):
            batch_idx = indices[i:i + batch_size]
            bx = x_train[batch_idx].to(device)
            by_act = y_act_tr[batch_idx].to(device)
            by_dan = y_dan_tr[batch_idx].to(device)
            by_saf = y_saf_tr[batch_idx].to(device)

            optimizer.zero_grad()
            out_act, out_dan, out_saf = model(bx)

            loss_a = crit_action(out_act, by_act)
            loss_d = crit_danger(out_dan, by_dan)
            loss_s = crit_safety(out_saf, by_saf)

            # Combined multi-task loss
            total_loss = loss_a + 0.5 * loss_d + 0.5 * loss_s
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / max(1, num_batches)

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            vx = x_val.to(device)
            v_act, v_dan, v_saf = model(vx)
            v_loss = (crit_action(v_act, y_act_va.to(device)) +
                      0.5 * crit_danger(v_dan, y_dan_va.to(device)) +
                      0.5 * crit_safety(v_saf, y_saf_va.to(device))).item()

            pred_act = torch.argmax(v_act, dim=-1)
            pred_dan = (torch.sigmoid(v_dan) > 0.5).float()
            pred_saf = torch.argmax(v_saf, dim=-1)

            acc_act = (pred_act == y_act_va.to(device)).float().mean().item()
            acc_dan = (pred_dan == y_dan_va.to(device)).float().mean().item()
            acc_saf = (pred_saf == y_saf_va.to(device)).float().mean().item()

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": v_loss,
            "action_acc": acc_act,
            "danger_acc": acc_dan,
            "safety_acc": acc_saf,
        })

    train_duration = time.perf_counter() - start_time

    # Display training progress table
    headers = ["Epoch", "Train Loss", "Val Loss", "Action Acc (Choice)", "Hazard Acc (Noul)", "Safety Acc (Score)"]
    rows = []
    for h in history:
        rows.append([
            f"Epoch {h['epoch']:02d}",
            f"{h['train_loss']:.4f}",
            f"{h['val_loss']:.4f}",
            f"[bold green]{h['action_acc']:.1%}[/bold green]",
            f"[bold green]{h['danger_acc']:.1%}[/bold green]",
            f"[bold green]{h['safety_acc']:.1%}[/bold green]",
        ])

    print_metrics_table(headers, rows, title=f"Laya Snake Fine-Tuning Convergence ({train_duration:.2f}s)")

    # Save fine-tuned checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "history": history,
        "base_model": "convaiinnovations/laya",
    }, save_path)
    print(f"\n[3] Saved Fine-Tuned Checkpoint to: {os.path.abspath(save_path)}")

    return {"history": history, "checkpoint": save_path, "device": device_str}


if __name__ == "__main__":
    train_snake_model()
