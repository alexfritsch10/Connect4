import numpy as np
from agents.common import BoardPiece, PlayerAction, apply_player_action, connected_four, NO_PLAYER, other_player


def evaluate_heuristic(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> int:
    """
    Calculates a score for a board

    Parameters
    ----------
    board : np.ndarray
            Board that the move is performed on
    action: PlayerAction
            Column of the move that is performed
    player: BoardPiece
            Player who performs the move

    Return
    ------
    Aggregated Score of all Moves that are possible after the action is performed

    """

    board_copy = board.copy()
    board_copy = apply_player_action(board_copy, action, player, False)

    heuristic = 0
    # check if player can win with this action
    if connected_four(board_copy, player, None):
        heuristic = 99
        return heuristic

    # check if other player can win with this action
    board_copy2 = board.copy()
    apply_player_action(board_copy2, action, other_player(player), False)

    if connected_four(board_copy2, other_player(player), None):
        heuristic = -99
        return heuristic

    # find lowest open row
    for row in range(6):
        if board[row, action] == NO_PLAYER:
            break
        if row == 5:
            raise ValueError("column can't be played")

    # initialize calculation values
    skip_a, skip_b, skip_c, skip_d, skip_e, skip_f, skip_g, skip_h = False, False, False, False, False, False, False, False
    streak_ab, streak_cd, streak_ef, streak_gh = 1, 1, 1, 1
    heuristic_a, heuristic_b, heuristic_c, heuristic_d, heuristic_e, heuristic_f, heuristic_g, heuristic_h = 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(1, 4):
        if (action+i) < 7 and not skip_a:
            if board[row, action+i] == player:
                heuristic_a += 1
                streak_ab += 1
            elif board[row, action+i] == NO_PLAYER:
                streak_ab += 1
            else:
                skip_a = True

        if (action-i) > -1 and not skip_b:
            if board[row, action-i] == player:
                heuristic_b += 1
                streak_ab += 1
            elif board[row, action-i] == NO_PLAYER:
                streak_ab += 1
            else:
                skip_b = True

        if (row+i) < 6 and not skip_c:
            if board[row+i, action] == player:
                heuristic_c += 1
                streak_cd += 1
            elif board[row+i, action] == NO_PLAYER:
                streak_cd += 1
            else:
                skip_c = True

        if (row-i) > -1 and not skip_d:
            if board[row-i, action] == player:
                heuristic_d += 1
                streak_cd += 1
            elif board[row-i, action] == NO_PLAYER:
                streak_cd += 1
            else:
                skip_d = True

        if ((action+i) < 7 and (row+i) < 6) and not skip_e:
            if board[row+i, action+i] == player:
                heuristic_e += 1
                streak_ef += 1
            elif board[row+i, action+i] == NO_PLAYER:
                streak_ef += 1
            else:
                skip_e = True

        if ((action-i) > -1 and (row-i) > -1) and not skip_f:
            if board[row-i, action-i] == player:
                heuristic_f += 1
                streak_ef += 1
            elif board[row-i, action-i] == NO_PLAYER:
                streak_ef += 1
            else:
                skip_f = True

        if ((action+i) < 7 and (row-i) > -1) and not skip_g:
            if board[row-i, action+i] == player:
                heuristic_g += 1
                streak_gh += 1
            elif board[row-i, action+i] == NO_PLAYER:
                streak_gh += 1
            else:
                skip_g = True

        if ((action-i) > -1 and (row+i) < 6) and not skip_h:
            if board[row+i, action-i] == player:
                heuristic_h += 1
                streak_gh += 1
            elif board[row+i, action-i] == NO_PLAYER:
                streak_gh += 1
            else:
                skip_h = True

    if streak_ab < 4:
        # wenn mit dem move in einer Reihe keine 4 erreicht werden können
        heuristic_a = 0
        heuristic_b = 0
    elif streak_ab == 7:
        heuristic += 2
    else:
        # (streak_ab > 3) and (streak_ab < 7):
        heuristic += 1

    if streak_cd < 4:
        # wenn mit dem move in einer Spalte keine 4 erreicht werden können
        heuristic_c = 0
        heuristic_d = 0
    elif streak_cd == 7:
        heuristic += 2
    else:
        # (streak_cd > 3) and (streak_cd < 7):
        heuristic += 1

    if streak_ef < 4:
        # wenn mit dem move in einer rechts-Diagonalen keine 4 erreicht werden können
        heuristic_e = 0
        heuristic_f = 0
    elif streak_ef == 7:
        heuristic += 2
    else:
        # (streak_ef > 3) and (streak_ef < 7):
        heuristic += 1

    if streak_gh < 4:
        # wenn mit dem move in einer links-Diagonalen keine 4 erreicht werden können
        heuristic_g = 0
        heuristic_h = 0
    elif streak_gh == 7:
        heuristic += 2
    else:
        # (streak_gh > 3) and (streak_gh < 7):
        heuristic += 1


    heuristic += heuristic_a + heuristic_b + heuristic_c + heuristic_d + heuristic_e + heuristic_f + heuristic_g + heuristic_h

    return heuristic
