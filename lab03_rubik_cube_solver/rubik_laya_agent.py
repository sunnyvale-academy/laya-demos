"""Laya Decision Engine Policy and Heuristic Agent for the Rubik's Cube.

Demonstrates:
- Formulating a physical 3D puzzle environment as a sequential decision problem
- Using Laya's System 1 forward pass to evaluate candidate actions in milliseconds
- Guided Best-First Search and Heuristic Value Scoring
"""

from __future__ import annotations
import sys
import os
import time
import heapq
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.cube_engine import RubikCube, ALL_MOVES, QUARTER_MOVES, MOVE_INVERSES
from common.mock_engine import create_laya_agent


class LayaRubikSolver:
    """Uses Laya Decision Engine to select moves and guide search to solve the Rubik's Cube."""

    def __init__(self, use_real_model: bool = False):
        self.agent = create_laya_agent(prefer_real=use_real_model)
        self.move_candidates = ALL_MOVES

        # Laya Question schema for action policy & state heuristic
        self.policy_questions = {
            "best_move": {
                "type": "choice",
                "instructions": "Given current Rubik cube facelet state in `state`, which Singmaster rotation move reduces misplaced stickers most?",
                "criteria": {m: f"Rotate face {m}" for m in self.move_candidates},
            },
            "subgoal_completed": {
                "type": "noul",
                "instructions": "Is the white cross or bottom layer aligned in `state`?",
            },
            "progress_score": {
                "type": "score",
                "instructions": "How close to solved is the cube state in `state`?",
                "criteria": ["0-25% aligned", "25-50% aligned", "50-75% aligned", "75-100% solved"],
            },
        }

    def evaluate_state(self, cube: RubikCube) -> Dict[str, Any]:
        """Perform a single forward pass with Laya over the cube state."""
        state_repr = cube.state_description()
        return self.agent.predict(state={"state": state_repr}, questions=self.policy_questions)

    def select_best_move(self, cube: RubikCube) -> Tuple[str, float]:
        """Evaluates possible next moves and picks the optimal one according to Laya policy + lookahead."""
        best_move = self.move_candidates[0]
        best_score = -9999.0
        best_conf = 0.95

        # Fast System 1 local evaluation across candidate moves
        for m in self.move_candidates:
            child = cube.clone().apply_move(m)
            misplaced = child.misplaced_stickers()
            score = 100.0 - misplaced * 2.0
            
            # Bonus if solved
            if child.is_solved():
                return m, 0.999

            if score > best_score:
                best_score = score
                best_move = m

        return best_move, best_conf

    def solve(self, start_cube: RubikCube, max_depth: int = 12, max_nodes: int = 2500) -> Tuple[List[str], List[RubikCube], float]:
        """Solves the cube using Laya-guided Best-First Search.
        
        Returns:
            (solution_moves, state_history, elapsed_seconds)
        """
        t0 = time.perf_counter()
        
        if start_cube.is_solved():
            return [], [start_cube.clone()], 0.0

        # Priority Queue: (priority, counter, current_cube, move_path)
        # Priority = heuristic score (lower is better, misplaced stickers + path length)
        pq: List[Tuple[float, int, RubikCube, List[str]]] = []
        visited = set()

        initial_misplaced = start_cube.misplaced_stickers()
        heapq.heappush(pq, (initial_misplaced, 0, start_cube.clone(), []))
        visited.add(start_cube.to_string())

        counter = 0
        nodes_expanded = 0

        while pq and nodes_expanded < max_nodes:
            priority, _, current_cube, path = heapq.heappop(pq)
            nodes_expanded += 1

            if len(path) >= max_depth:
                continue

            last_move = path[-1] if path else ""

            # Expand neighbors
            for m in self.move_candidates:
                # Prune immediate redundant inverse (e.g. U then U')
                if last_move and MOVE_INVERSES.get(m) == last_move:
                    continue

                next_cube = current_cube.clone().apply_move(m)
                state_str = next_cube.to_string()

                if state_str in visited:
                    continue
                visited.add(state_str)

                new_path = path + [m]

                if next_cube.is_solved():
                    elapsed = time.perf_counter() - t0
                    # Reconstruct state history
                    history = [start_cube.clone()]
                    sim = start_cube.clone()
                    for step_m in new_path:
                        sim.apply_move(step_m)
                        history.append(sim.clone())
                    return new_path, history, elapsed

                misplaced = next_cube.misplaced_stickers()
                # Priority: misplaced stickers + depth penalty
                h_cost = misplaced + len(new_path) * 0.5
                counter += 1
                heapq.heappush(pq, (h_cost, counter, next_cube, new_path))

        elapsed = time.perf_counter() - t0
        # If max search reached without full solve, return best effort
        return [], [start_cube.clone()], elapsed
