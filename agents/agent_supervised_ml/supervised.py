import pickle
import numpy as np

from agents.agent_supervised_ml.classification import logistic_regression, multilayer_perceptron
from agents.common import BoardPiece, PlayerAction, SavedState, move_is_possible, PLAYER1, PLAYER2, \
    column_to_be_played_for_win, other_player
from typing import Optional, Tuple
import random

MODEL = "logistic_regression.pickle"


def predict_move(board: np.ndarray) -> int:
    """

    :param board:
    :return:
    """
    b = np.ndarray((0, 42), int)
    b = np.vstack([b, board.flatten()])

    pickle_in = open(MODEL, "rb")
    clf = pickle.load(pickle_in)
    action = clf.predict(b)

    return action - 1


def generate_move_supervised(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    # Choose a valid, non-full column with supervised ml algorithm and return it as `action`

    # if agent can win
    win = column_to_be_played_for_win(board, player)
    if win > -1:
        return win, saved_state

    # if agent has to block
    block = column_to_be_played_for_win(board, other_player(player))
    if block > -1:
        return block, saved_state

    action = np.int8(predict_move(board))

    if move_is_possible(board, action):
        return action, saved_state
    else:
        print('prediction error')
        possible_moves = []
        if move_is_possible(board, np.int8(3)):
            return np.int8(3), saved_state
        else:
            for i in range(7):
                if move_is_possible(board, np.int8(i)):
                    possible_moves = np.append(possible_moves, i)

    action = np.random.choice(possible_moves, 1)
    action = action.astype(np.int8)

    return action, saved_state
