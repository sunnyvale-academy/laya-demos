"""Rubik's Cube Simulator, Representation, and Visualizer.

Supports standard Singmaster notation:
- Faces: U (Up), D (Down), F (Front), B (Back), L (Left), R (Right)
- Moves: U, U', U2, D, D', D2, F, F', F2, B, B', B2, L, L', L2, R, R', R2
- Colors: White (U), Yellow (D), Green (F), Blue (B), Orange (L), Red (R)
"""

from __future__ import annotations
import copy
import random
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Color(str, Enum):
    WHITE = "W"
    YELLOW = "Y"
    GREEN = "G"
    BLUE = "B"
    ORANGE = "O"
    RED = "R"


class Face(str, Enum):
    U = "U"  # Up (White)
    D = "D"  # Down (Yellow)
    F = "F"  # Front (Green)
    B = "B"  # Back (Blue)
    L = "L"  # Left (Orange)
    R = "R"  # Right (Red)


# Color mapping for ANSI rendering in terminal
ANSI_COLORS = {
    Color.WHITE: "\033[47m\033[30m W \033[0m",
    Color.YELLOW: "\033[43m\033[30m Y \033[0m",
    Color.GREEN: "\033[42m\033[30m G \033[0m",
    Color.BLUE: "\033[44m\033[37m B \033[0m",
    Color.ORANGE: "\033[48;5;208m\033[30m O \033[0m",
    Color.RED: "\033[41m\033[37m R \033[0m",
}

# Standard 18 Singmaster moves
ALL_MOVES = [
    "U", "U'", "U2",
    "D", "D'", "D2",
    "F", "F'", "F2",
    "B", "B'", "B2",
    "L", "L'", "L2",
    "R", "R'", "R2"
]

# Standard 12 atomic quarter-turn moves
QUARTER_MOVES = ["U", "U'", "D", "D'", "F", "F'", "B", "B'", "L", "L'", "R", "R'"]

MOVE_INVERSES = {
    "U": "U'", "U'": "U", "U2": "U2",
    "D": "D'", "D'": "D", "D2": "D2",
    "F": "F'", "F'": "F", "F2": "F2",
    "B": "B'", "B'": "B", "B2": "B2",
    "L": "L'", "L'": "L", "L2": "L2",
    "R": "R'", "R'": "R", "R2": "R2",
}


class Move(str, Enum):
    U = "U"
    U_PRIME = "U'"
    U2 = "U2"
    D = "D"
    D_PRIME = "D'"
    D2 = "D2"
    F = "F"
    F_PRIME = "F'"
    F2 = "F2"
    B = "B"
    B_PRIME = "B'"
    B2 = "B2"
    L = "L"
    L_PRIME = "L'"
    L2 = "L2"
    R = "R"
    R_PRIME = "R'"
    R2 = "R2"


class RubikCube:
    """Represents a 3x3x3 Rubik's Cube state with 54 facelets.

    Faces indexing layout (3x3 grid for each face):
      0 1 2
      3 4 5
      6 7 8
    """

    def __init__(self, state_dict: Optional[Dict[Face, List[Color]]] = None):
        if state_dict is not None:
            self.faces = {face: list(colors) for face, colors in state_dict.items()}
        else:
            self.reset()

    def reset(self) -> None:
        """Reset the cube to the canonical solved state."""
        self.faces = {
            Face.U: [Color.WHITE] * 9,
            Face.D: [Color.YELLOW] * 9,
            Face.F: [Color.GREEN] * 9,
            Face.B: [Color.BLUE] * 9,
            Face.L: [Color.ORANGE] * 9,
            Face.R: [Color.RED] * 9,
        }

    def clone(self) -> RubikCube:
        """Create a deep copy of the current cube state."""
        return RubikCube(self.faces)

    def is_solved(self) -> bool:
        """Check if all 6 faces have uniform colors."""
        for face, colors in self.faces.items():
            first = colors[0]
            if any(c != first for c in colors):
                return False
        return True

    def misplaced_stickers(self) -> int:
        """Return count of stickers that do not match their face center sticker."""
        count = 0
        for colors in self.faces.values():
            center = colors[4]
            count += sum(1 for c in colors if c != center)
        return count

    def stage_completion_score(self) -> float:
        """Calculates a normalized score in [0.0, 1.0] reflecting solved progress.
        
        1.0 means fully solved; 0.0 means maximally scrambled.
        """
        misplaced = self.misplaced_stickers()
        # Max misplaced is 48 (center stickers are fixed)
        return max(0.0, 1.0 - (misplaced / 48.0))

    def _rotate_face_clockwise(self, face: Face) -> None:
        """Rotate the 3x3 matrix of a single face 90 degrees clockwise."""
        f = self.faces[face]
        self.faces[face] = [
            f[6], f[3], f[0],
            f[7], f[4], f[1],
            f[8], f[5], f[2]
        ]

    def _rotate_face_counter_clockwise(self, face: Face) -> None:
        """Rotate the 3x3 matrix of a single face 90 degrees counter-clockwise."""
        f = self.faces[face]
        self.faces[face] = [
            f[2], f[5], f[8],
            f[1], f[4], f[7],
            f[0], f[3], f[6]
        ]

    def apply_move(self, move: str) -> RubikCube:
        """Apply a Singmaster move to the cube in-place and return self."""
        move = move.strip()

        if move == "U":
            self._rotate_face_clockwise(Face.U)
            # Cycle adjacent edges: F -> L -> B -> R -> F (top rows)
            temp = self.faces[Face.F][0:3]
            self.faces[Face.F][0:3] = self.faces[Face.R][0:3]
            self.faces[Face.R][0:3] = self.faces[Face.B][0:3]
            self.faces[Face.B][0:3] = self.faces[Face.L][0:3]
            self.faces[Face.L][0:3] = temp

        elif move == "U'":
            self._rotate_face_counter_clockwise(Face.U)
            temp = self.faces[Face.F][0:3]
            self.faces[Face.F][0:3] = self.faces[Face.L][0:3]
            self.faces[Face.L][0:3] = self.faces[Face.B][0:3]
            self.faces[Face.B][0:3] = self.faces[Face.R][0:3]
            self.faces[Face.R][0:3] = temp

        elif move == "U2":
            self.apply_move("U")
            self.apply_move("U")

        elif move == "D":
            self._rotate_face_clockwise(Face.D)
            # Cycle adjacent edges: F -> R -> B -> L -> F (bottom rows)
            temp = self.faces[Face.F][6:9]
            self.faces[Face.F][6:9] = self.faces[Face.L][6:9]
            self.faces[Face.L][6:9] = self.faces[Face.B][6:9]
            self.faces[Face.B][6:9] = self.faces[Face.R][6:9]
            self.faces[Face.R][6:9] = temp

        elif move == "D'":
            self._rotate_face_counter_clockwise(Face.D)
            temp = self.faces[Face.F][6:9]
            self.faces[Face.F][6:9] = self.faces[Face.R][6:9]
            self.faces[Face.R][6:9] = self.faces[Face.B][6:9]
            self.faces[Face.B][6:9] = self.faces[Face.L][6:9]
            self.faces[Face.L][6:9] = temp

        elif move == "D2":
            self.apply_move("D")
            self.apply_move("D")

        elif move == "R":
            self._rotate_face_clockwise(Face.R)
            # Cycle U -> F -> D -> B (right columns)
            u2, u5, u8 = self.faces[Face.U][2], self.faces[Face.U][5], self.faces[Face.U][8]
            f2, f5, f8 = self.faces[Face.F][2], self.faces[Face.F][5], self.faces[Face.F][8]
            d2, d5, d8 = self.faces[Face.D][2], self.faces[Face.D][5], self.faces[Face.D][8]
            b0, b3, b6 = self.faces[Face.B][0], self.faces[Face.B][3], self.faces[Face.B][6]

            self.faces[Face.U][2], self.faces[Face.U][5], self.faces[Face.U][8] = f2, f5, f8
            self.faces[Face.F][2], self.faces[Face.F][5], self.faces[Face.F][8] = d2, d5, d8
            self.faces[Face.D][2], self.faces[Face.D][5], self.faces[Face.D][8] = b6, b3, b0
            self.faces[Face.B][0], self.faces[Face.B][3], self.faces[Face.B][6] = u8, u5, u2

        elif move == "R'":
            self._rotate_face_counter_clockwise(Face.R)
            u2, u5, u8 = self.faces[Face.U][2], self.faces[Face.U][5], self.faces[Face.U][8]
            f2, f5, f8 = self.faces[Face.F][2], self.faces[Face.F][5], self.faces[Face.F][8]
            d2, d5, d8 = self.faces[Face.D][2], self.faces[Face.D][5], self.faces[Face.D][8]
            b0, b3, b6 = self.faces[Face.B][0], self.faces[Face.B][3], self.faces[Face.B][6]

            self.faces[Face.U][2], self.faces[Face.U][5], self.faces[Face.U][8] = b6, b3, b0
            self.faces[Face.B][0], self.faces[Face.B][3], self.faces[Face.B][6] = d8, d5, d2
            self.faces[Face.D][2], self.faces[Face.D][5], self.faces[Face.D][8] = f2, f5, f8
            self.faces[Face.F][2], self.faces[Face.F][5], self.faces[Face.F][8] = u2, u5, u8

        elif move == "R2":
            self.apply_move("R")
            self.apply_move("R")

        elif move == "L":
            self._rotate_face_clockwise(Face.L)
            # Cycle U -> B -> D -> F (left columns)
            u0, u3, u6 = self.faces[Face.U][0], self.faces[Face.U][3], self.faces[Face.U][6]
            b8, b5, b2 = self.faces[Face.B][8], self.faces[Face.B][5], self.faces[Face.B][2]
            d0, d3, d6 = self.faces[Face.D][0], self.faces[Face.D][3], self.faces[Face.D][6]
            f0, f3, f6 = self.faces[Face.F][0], self.faces[Face.F][3], self.faces[Face.F][6]

            self.faces[Face.U][0], self.faces[Face.U][3], self.faces[Face.U][6] = b8, b5, b2
            self.faces[Face.B][8], self.faces[Face.B][5], self.faces[Face.B][2] = d0, d3, d6
            self.faces[Face.D][0], self.faces[Face.D][3], self.faces[Face.D][6] = f0, f3, f6
            self.faces[Face.F][0], self.faces[Face.F][3], self.faces[Face.F][6] = u0, u3, u6

        elif move == "L'":
            self._rotate_face_counter_clockwise(Face.L)
            u0, u3, u6 = self.faces[Face.U][0], self.faces[Face.U][3], self.faces[Face.U][6]
            b8, b5, b2 = self.faces[Face.B][8], self.faces[Face.B][5], self.faces[Face.B][2]
            d0, d3, d6 = self.faces[Face.D][0], self.faces[Face.D][3], self.faces[Face.D][6]
            f0, f3, f6 = self.faces[Face.F][0], self.faces[Face.F][3], self.faces[Face.F][6]

            self.faces[Face.U][0], self.faces[Face.U][3], self.faces[Face.U][6] = f0, f3, f6
            self.faces[Face.F][0], self.faces[Face.F][3], self.faces[Face.F][6] = d0, d3, d6
            self.faces[Face.D][0], self.faces[Face.D][3], self.faces[Face.D][6] = b8, b5, b2
            self.faces[Face.B][8], self.faces[Face.B][5], self.faces[Face.B][2] = u0, u3, u6

        elif move == "L2":
            self.apply_move("L")
            self.apply_move("L")

        elif move == "F":
            self._rotate_face_clockwise(Face.F)
            u6, u7, u8 = self.faces[Face.U][6], self.faces[Face.U][7], self.faces[Face.U][8]
            r0, r3, r6 = self.faces[Face.R][0], self.faces[Face.R][3], self.faces[Face.R][6]
            d2, d1, d0 = self.faces[Face.D][2], self.faces[Face.D][1], self.faces[Face.D][0]
            l8, l5, l2 = self.faces[Face.L][8], self.faces[Face.L][5], self.faces[Face.L][2]

            self.faces[Face.U][6], self.faces[Face.U][7], self.faces[Face.U][8] = l8, l5, l2
            self.faces[Face.R][0], self.faces[Face.R][3], self.faces[Face.R][6] = u6, u7, u8
            self.faces[Face.D][2], self.faces[Face.D][1], self.faces[Face.D][0] = r0, r3, r6
            self.faces[Face.L][8], self.faces[Face.L][5], self.faces[Face.L][2] = d2, d1, d0

        elif move == "F'":
            self._rotate_face_counter_clockwise(Face.F)
            u6, u7, u8 = self.faces[Face.U][6], self.faces[Face.U][7], self.faces[Face.U][8]
            r0, r3, r6 = self.faces[Face.R][0], self.faces[Face.R][3], self.faces[Face.R][6]
            d2, d1, d0 = self.faces[Face.D][2], self.faces[Face.D][1], self.faces[Face.D][0]
            l8, l5, l2 = self.faces[Face.L][8], self.faces[Face.L][5], self.faces[Face.L][2]

            self.faces[Face.U][6], self.faces[Face.U][7], self.faces[Face.U][8] = r0, r3, r6
            self.faces[Face.L][8], self.faces[Face.L][5], self.faces[Face.L][2] = u6, u7, u8
            self.faces[Face.D][2], self.faces[Face.D][1], self.faces[Face.D][0] = l8, l5, l2
            self.faces[Face.R][0], self.faces[Face.R][3], self.faces[Face.R][6] = d2, d1, d0

        elif move == "F2":
            self.apply_move("F")
            self.apply_move("F")

        elif move == "B":
            self._rotate_face_clockwise(Face.B)
            u2, u1, u0 = self.faces[Face.U][2], self.faces[Face.U][1], self.faces[Face.U][0]
            l0, l3, l6 = self.faces[Face.L][0], self.faces[Face.L][3], self.faces[Face.L][6]
            d6, d7, d8 = self.faces[Face.D][6], self.faces[Face.D][7], self.faces[Face.D][8]
            r8, r5, r2 = self.faces[Face.R][8], self.faces[Face.R][5], self.faces[Face.R][2]

            self.faces[Face.U][2], self.faces[Face.U][1], self.faces[Face.U][0] = r8, r5, r2
            self.faces[Face.L][0], self.faces[Face.L][3], self.faces[Face.L][6] = u2, u1, u0
            self.faces[Face.D][6], self.faces[Face.D][7], self.faces[Face.D][8] = l0, l3, l6
            self.faces[Face.R][8], self.faces[Face.R][5], self.faces[Face.R][2] = d6, d7, d8

        elif move == "B'":
            self._rotate_face_counter_clockwise(Face.B)
            u2, u1, u0 = self.faces[Face.U][2], self.faces[Face.U][1], self.faces[Face.U][0]
            l0, l3, l6 = self.faces[Face.L][0], self.faces[Face.L][3], self.faces[Face.L][6]
            d6, d7, d8 = self.faces[Face.D][6], self.faces[Face.D][7], self.faces[Face.D][8]
            r8, r5, r2 = self.faces[Face.R][8], self.faces[Face.R][5], self.faces[Face.R][2]

            self.faces[Face.U][2], self.faces[Face.U][1], self.faces[Face.U][0] = l0, l3, l6
            self.faces[Face.R][8], self.faces[Face.R][5], self.faces[Face.R][2] = u2, u1, u0
            self.faces[Face.D][6], self.faces[Face.D][7], self.faces[Face.D][8] = r8, r5, r2
            self.faces[Face.L][0], self.faces[Face.L][3], self.faces[Face.L][6] = d6, d7, d8

        elif move == "B2":
            self.apply_move("B")
            self.apply_move("B")

        else:
            raise ValueError(f"Unknown move: {move}")

        return self

    def apply_moves(self, moves: List[str] | str) -> RubikCube:
        """Apply a sequence of moves (e.g. 'R U R\\' or ['R', 'U', 'R\\''])."""
        if isinstance(moves, str):
            move_list = moves.strip().split()
        else:
            move_list = moves
        for m in move_list:
            if m:
                self.apply_move(m)
        return self

    def scramble(self, steps: int = 15, seed: Optional[int] = None) -> List[str]:
        """Apply a random sequence of moves to scramble the cube."""
        if seed is not None:
            random.seed(seed)
        scramble_moves: List[str] = []
        last_face = ""
        while len(scramble_moves) < steps:
            m = random.choice(ALL_MOVES)
            face = m[0]
            if face == last_face:
                continue
            scramble_moves.append(m)
            self.apply_move(m)
            last_face = face
        return scramble_moves

    def to_string(self) -> str:
        """Serialize facelet state to a compact 54-char string: U-D-F-B-L-R."""
        return "".join(
            "".join(c.value for c in self.faces[f])
            for f in [Face.U, Face.D, Face.F, Face.B, Face.L, Face.R]
        )

    def state_description(self) -> str:
        """Generates a structured natural text description of the cube state for Laya."""
        lines = []
        for face in [Face.U, Face.D, Face.F, Face.B, Face.L, Face.R]:
            f_colors = "".join(c.value for c in self.faces[face])
            lines.append(f"{face.value}:[{f_colors[:3]}|{f_colors[3:6]}|{f_colors[6:]}]")
        misplaced = self.misplaced_stickers()
        lines.append(f"MisplacedCount:{misplaced}")
        return " ".join(lines)

    def render_ascii(self, use_color: bool = True) -> str:
        """Render a 2D unfolded cross layout of the cube in terminal."""
        def row_str(face: Face, row_idx: int) -> str:
            start = row_idx * 3
            cells = self.faces[face][start:start+3]
            if use_color:
                return "".join(ANSI_COLORS.get(c, str(c.value)) for c in cells)
            else:
                return " ".join(c.value for c in cells)

        empty_pad = "         " if use_color else "      "
        lines = []
        # U face (top)
        for r in range(3):
            lines.append(f"{empty_pad}{row_str(Face.U, r)}")
        # Middle row: L, F, R, B faces
        for r in range(3):
            line = (
                f"{row_str(Face.L, r)}"
                f"{row_str(Face.F, r)}"
                f"{row_str(Face.R, r)}"
                f"{row_str(Face.B, r)}"
            )
            lines.append(line)
        # D face (bottom)
        for r in range(3):
            lines.append(f"{empty_pad}{row_str(Face.D, r)}")

        return "\n".join(lines)
