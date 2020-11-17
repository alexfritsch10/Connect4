import numpy as np
from agents.common import BoardPiece, PlayerAction, SavedState
from typing import Optional, Tuple
import random


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column randomly and return it as `action`
    topRow = board[5, :]
    idxList = []
    for idx, col in enumerate(topRow):
        if col == 0:
            idxList.append(idx)
    action = PlayerAction(random.choice(idxList))
    return action, saved_state
