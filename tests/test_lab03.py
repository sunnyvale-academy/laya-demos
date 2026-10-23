import pytest
from common.cube_engine import RubikCube, Color, Face
try:
    from labs.lab03_rubik_cube_solver.rubik_laya_agent import LayaRubikSolver
except ImportError:
    from lab03_rubik_cube_solver.rubik_laya_agent import LayaRubikSolver


def test_rubik_cube_solved():
    cube = RubikCube()
    assert cube.is_solved() is True
    assert cube.misplaced_stickers() == 0
    assert cube.stage_completion_score() == 1.0


def test_rubik_cube_moves_and_scramble():
    cube = RubikCube()
    cube.apply_move("R")
    assert cube.is_solved() is False
    assert cube.misplaced_stickers() > 0
    cube.apply_move("R'")
    assert cube.is_solved() is True

    moves = cube.scramble(steps=4, seed=42)
    assert len(moves) == 4
    assert cube.is_solved() is False


def test_rubik_laya_solver():
    cube = RubikCube()
    cube.scramble(steps=3, seed=123)
    solver = LayaRubikSolver()
    solution, history, elapsed = solver.solve(cube, max_depth=6)
    assert len(solution) > 0
    assert elapsed >= 0.0

    # Verify solution actually solves the cube
    test_cube = cube.clone()
    test_cube.apply_moves(solution)
    assert test_cube.is_solved() is True
