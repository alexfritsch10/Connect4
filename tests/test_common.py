import numpy as np
from agents.common import BoardPiece, NO_PLAYER


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)
    print(ret)


def test_string_to_board():
    from agents.common import string_to_board
    boardString = "[=============]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 1 0 0 0 0 0]\n[0 1 1 0 2 2 0]\n[=============]\n[0 1 2 3 4 5 6]"
    boardArray = string_to_board(boardString)

    assert boardArray.shape == (6,7)
    assert boardArray[0][1] == 1
    assert boardArray[0][5] == 0
    assert boardArray[4][3] == 0

def test_apply_player_action():
    from agents.common import apply_player_action, string_to_board

    boardString = "[=============]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 0 0 0 0 0 0]\n[0 1 0 0 0 0 0]\n[0 1 1 0 2 2 0]\n[=============]\n[0 1 2 3 4 5 6]"
    boardArray = string_to_board(boardString)

    assert boardArray[1][2] == 0

    newBoard = apply_player_action(boardArray, 1, 1)

    assert newBoard[1][2] == 1

def test_connected_four():
    from agents.common import connected_four, string_to_board

    boardString = "[=============]\n[0 0 0 2 0 0 0]\n[0 0 0 1 0 0 0]\n[0 0 1 1 0 0 0]\n[0 0 2 1 0 0 0]\n[2 2 2 2 0 0 2]\n[1 1 2 1 0 0 1]\n[=============]\n[0 1 2 3 4 5 6]"
    boardArray = string_to_board(boardString)

    assert connected_four(boardArray,2) == True


def test_check_end_state():
    from agents.common import check_end_state, string_to_board, GameState

    boardString = "[=============]\n[0 0 0 2 0 0 0]\n[0 0 0 1 0 0 0]\n[0 0 1 1 0 0 0]\n[0 0 2 1 0 0 0]\n[2 2 2 2 0 0 2]\n[1 1 2 1 0 0 1]\n[=============]\n[0 1 2 3 4 5 6]"
    boardArray = string_to_board(boardString)

    assert check_end_state(boardArray,2) == GameState.IS_WIN