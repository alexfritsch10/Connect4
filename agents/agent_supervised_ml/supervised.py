import numpy as np

from agents.agent_supervised_ml.classification import logistic_regression
from agents.common import BoardPiece, PlayerAction, SavedState, move_is_possible, PLAYER1, PLAYER2
from typing import Optional, Tuple
import random


def board_to_move_seq(board: np.ndarray) -> list:

    move_sequence = []
    player_1 = []
    player_2 = []
    last = PLAYER2

    for i in range(6):
        for j in range(7):
            if board[i, j] == PLAYER1:
                if last == PLAYER1:
                    player_1.append(j + 1)
                else:
                    move_sequence.append(j + 1)
                    last = PLAYER1
                    if player_2:
                        move_sequence.append(player_2.pop(0))
                        last = PLAYER2
            elif board[i, j] == PLAYER2:
                if last == PLAYER2:
                    player_2.append(j + 1)
                else:
                    move_sequence.append(j+1)
                    last = PLAYER2
                    if player_1:
                        move_sequence.append(player_1.pop(0))
                        last = PLAYER1


    print(player_1)
    print(player_2)

    return move_sequence


def move_to_prediction(action: PlayerAction, seq: np.ndarray) -> int:
    """
    get the predicted score for a given sequence of moves and the possible next move
    """
    move = np.ndarray((0, 42), int)
    seq = np.append(seq, action + 1)
    print(seq)
    move = np.append(move, seq)
    print('move: ', move)

    for i in range(42 - len(move)):  # 41-len(moves) 0en müssen hinzugefügt werden
        move = np.append(move, 0)

    score = logistic_regression(move)
    return score


def generate_move_supervised(
    board: np.ndarray, player: BoardPiece, seq: Optional[list], saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column with supervised ml algorithm and return it as `action`
    min = np.Inf

    for i in range(6):
        if move_is_possible(board, i):
            current = move_to_prediction(i, seq)
            if current < min:
                min = current
                action = i


    return action, saved_state