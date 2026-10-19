"""Nokia 3310 Snake Environment simulator with ANSI graphics and spatial state serialization."""

from __future__ import annotations
import copy
from enum import Enum
import random
from typing import Dict, List, Optional, Tuple, Any
from rich.console import Console
from rich.panel import Panel

console = Console()


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


# Reverse direction mapping (Snake cannot immediately reverse into its neck)
OPPOSITE_DIRECTIONS = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

# Coordinate delta offsets
DIRECTION_DELTAS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


class SnakeGame:
    """Nokia 3310 Snake Game Environment (Grid: 10x10 standard)."""

    def __init__(self, width: int = 10, height: int = 10, seed: Optional[int] = None):
        self.width = width
        self.height = height
        if seed is not None:
            random.seed(seed)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Resets the game to initial state."""
        start_x = self.width // 2
        start_y = self.height // 2
        # Head is index 0
        self.snake: List[Tuple[int, int]] = [
            (start_x, start_y),
            (start_x - 1, start_y),
            (start_x - 2, start_y),
        ]
        self.direction = Direction.RIGHT
        self.score = 0
        self.steps_alive = 0
        self.game_over = False
        self.food = self._spawn_food()
        return self.get_state_dict()

    def _spawn_food(self) -> Tuple[int, int]:
        """Spawns food at an empty grid location."""
        occupied = set(self.snake)
        available = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not available:
            return (-1, -1)  # Win condition (grid full)
        return random.choice(available)

    def is_collision(self, pt: Tuple[int, int]) -> bool:
        """Checks wall or self collision."""
        x, y = pt
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if pt in self.snake[:-1]:  # Tail moves unless eating, but check body
            return True
        return False

    def get_danger_profile(self) -> Dict[str, bool]:
        """Returns boolean danger flags for straight, left, and right relative to current direction."""
        head_x, head_y = self.snake[0]
        
        # Directions ordered clockwise: UP, RIGHT, DOWN, LEFT
        clock = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
        idx = clock.index(self.direction)
        dir_straight = self.direction
        dir_right = clock[(idx + 1) % 4]
        dir_left = clock[(idx - 1) % 4]

        dx_s, dy_s = DIRECTION_DELTAS[dir_straight]
        dx_r, dy_r = DIRECTION_DELTAS[dir_right]
        dx_l, dy_l = DIRECTION_DELTAS[dir_left]

        return {
            "danger_straight": self.is_collision((head_x + dx_s, head_y + dy_s)),
            "danger_right": self.is_collision((head_x + dx_r, head_y + dy_r)),
            "danger_left": self.is_collision((head_x + dx_l, head_y + dy_l)),
            "food_up": self.food[1] < head_y,
            "food_down": self.food[1] > head_y,
            "food_left": self.food[0] < head_x,
            "food_right": self.food[0] > head_x,
        }

    def get_expert_action(self) -> Direction:
        """Heuristic expert solver (A* shortest path with fallback survival move)."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        valid_moves: List[Tuple[Direction, int]] = []
        for d in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            if d == OPPOSITE_DIRECTIONS[self.direction]:
                continue
            dx, dy = DIRECTION_DELTAS[d]
            nxt = (head_x + dx, head_y + dy)
            if not self.is_collision(nxt):
                # Manhattan distance to food
                dist = abs(nxt[0] - food_x) + abs(nxt[1] - food_y)
                valid_moves.append((d, dist))

        if not valid_moves:
            return self.direction  # Trapped

        # Sort by shortest distance to food
        valid_moves.sort(key=lambda x: x[1])
        return valid_moves[0][0]

    def step(self, action: Direction) -> Tuple[Dict[str, Any], float, bool]:
        """Applies an action move to the environment."""
        if self.game_over:
            return self.get_state_dict(), 0.0, True

        # Prevent 180-degree instant reversal
        if action != OPPOSITE_DIRECTIONS[self.direction]:
            self.direction = action

        dx, dy = DIRECTION_DELTAS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        self.steps_alive += 1

        # Check collision
        if self.is_collision(new_head):
            self.game_over = True
            return self.get_state_dict(), -10.0, True

        self.snake.insert(0, new_head)
        reward = 0.0

        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._spawn_food()
            if self.food == (-1, -1):  # Board cleared
                self.game_over = True
                return self.get_state_dict(), 50.0, True
        else:
            self.snake.pop()
            # Small step reward/penalty based on distance to food
            prev_dist = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            reward = 0.1 if new_dist < prev_dist else -0.15

        # Cap max steps to prevent infinite loops
        if self.steps_alive > self.width * self.height * 4:
            self.game_over = True

        return self.get_state_dict(), reward, self.game_over

    def get_state_dict(self) -> Dict[str, Any]:
        """Constructs state dictionary for Laya Decision Engine."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        danger = self.get_danger_profile()

        # Build clear spatial text descriptor for Laya
        state_text = (
            f"Snake head at ({head_x}, {head_y}), moving {self.direction.value}. "
            f"Food is located at ({food_x}, {food_y}). "
            f"Obstacle in front: {danger['danger_straight']}, "
            f"Obstacle on right: {danger['danger_right']}, "
            f"Obstacle on left: {danger['danger_left']}. "
            f"Food direction relative to head: "
            f"{'NORTH ' if danger['food_up'] else ''}"
            f"{'SOUTH ' if danger['food_down'] else ''}"
            f"{'EAST ' if danger['food_right'] else ''}"
            f"{'WEST ' if danger['food_left'] else ''}."
        )

        return {
            "state_text": state_text,
            "head": (head_x, head_y),
            "food": (food_x, food_y),
            "direction": self.direction.value,
            "score": self.score,
            "length": len(self.snake),
            "danger": danger,
        }

    def render_nokia(self, use_color: bool = True) -> str:
        """Renders the game board in iconic Nokia 3310 monochrome/green retro styling."""
        lines = []
        border_top = "╔" + "═" * (self.width * 2) + "╗"
        border_bottom = "╚" + "═" * (self.width * 2) + "╝"

        lines.append(border_top)
        for y in range(self.height):
            row = ["║"]
            for x in range(self.width):
                pt = (x, y)
                if pt == self.snake[0]:
                    # Snake Head
                    row.append("🟩" if use_color else "▣ ")
                elif pt in self.snake[1:]:
                    # Snake Body
                    row.append("🟢" if use_color else "■ ")
                elif pt == self.food:
                    # Food / Bug
                    row.append("🍎" if use_color else "★ ")
                else:
                    row.append("  ")
            row.append("║")
            lines.append("".join(row))
        lines.append(border_bottom)
        lines.append(f" Nokia 3310 Snake | Score: {self.score:03d} | Length: {len(self.snake)} | Moves: {self.steps_alive}")
        return "\n".join(lines)


def get_snake_decision_questions() -> Dict[str, Dict[str, Any]]:
    """Defines the Laya Question Schema for Snake Policy & Safety."""
    return {
        "action": {
            "type": "choice",
            "instructions": "Select the best next move direction for the snake to reach food safely.",
            "criteria": {
                "UP": "move North if food is North and obstacle is not North",
                "DOWN": "move South if food is South and obstacle is not South",
                "LEFT": "move West if food is West and obstacle is not West",
                "RIGHT": "move East if food is East and obstacle is not East",
            },
        },
        "is_danger_ahead": {
            "type": "noul",
            "instructions": "Is there an immediate wall or body collision hazard directly ahead in the current direction?",
        },
        "path_safety_score": {
            "type": "score",
            "instructions": "Rate the overall safety of available paths (0: trapped/dead end, 1: tight, 2: open space, 3: optimal direct clear path).",
            "criteria": ["trapped", "tight", "open", "optimal"],
        },
    }
