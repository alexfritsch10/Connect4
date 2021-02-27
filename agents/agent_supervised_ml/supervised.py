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
    opens saved model to predict the move for the given board

    Parameters
    ----------
    board : np.ndarray
            board for which the next optimal move should be predicted

    Returns
    -------
    The move which is optimal in the current GameState
    """

    # convert board to model format
    b = np.ndarray((0, 42), int)
    b = np.vstack([b, board.flatten()])

    # load model and predict optimal move
    pickle_in = open(MODEL, "rb")
    clf = pickle.load(pickle_in)
    action = clf.predict(b)

    return action - 1


def generate_move_supervised(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Choose a valid, non-full column with the prediction of a supervised ml trained model, checks for possibility to win
    or requirement to block first, predict and check if the predicted move is actually possible, if not do middle column
    and if this is also not possible pick a random of the possible moves.

    Parameters
    ----------
    board : np.ndarray
            Board for which the optimal next move should be predicted
    player: BoardPiece
            Player for whom the optimal next move should be predicted
    saved_state : Optional[SavedState]
            Debugging param

    Returns
    -------
    Optimal Move for the current board and player, and saved_state
    """

    # if player can win
    win = column_to_be_played_for_win(board, player)
    if win > -1:
        return win, saved_state

    # if player has to block
    block = column_to_be_played_for_win(board, other_player(player))
    if block > -1:
        return block, saved_state

    action = PlayerAction(predict_move(board))

    # returns prediction if move is possible
    if move_is_possible(board, action):
        return action, saved_state
    # returns other move
    else:
        print('prediction error')
        # returns middle column if possible
        if move_is_possible(board, PlayerAction(3)):
            return PlayerAction(3), saved_state
        # returns random move from possible moves
        else:
            possible_moves = []
            for i in range(7):
                if move_is_possible(board, PlayerAction(i)):
                    possible_moves = np.append(possible_moves, i)

    action = np.random.choice(possible_moves, 1)
    action = action.astype(PlayerAction)

    return action, saved_state
