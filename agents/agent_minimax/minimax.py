import numpy as np
from agents.common import BoardPiece, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, SavedState, \
    Optional, Tuple, NO_PLAYER, column_to_be_played_for_win, initialize_game_state
from agents.agent_minimax.gameState import GameState


def evaluate_heuristic(board: np.ndarray, action: PlayerAction, player: BoardPiece) -> int:
    board_copy = board.copy()
    board_copy = apply_player_action(board_copy, action, player, False)

    heuristic = 0
    if connected_four(board_copy, player, None):
        heuristic = 99
        return heuristic

    # find lowest open row
    for row in range(6):
        if board[row, action] == NO_PLAYER:
            break
        if row == 5:
            raise ValueError("column can't be played")
                                                            # evaluate move at board[row, action]
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

    if streak_ab < 4:               # wenn mit dem move in einer Reihe keine 4 erreicht werden können
        heuristic_a = 0
        heuristic_b = 0
    elif streak_ab == 7:
        heuristic += 2
    else:                           # (streak_ab > 3) and (streak_ab < 7):
        heuristic += 1

    if streak_cd < 4:               # wenn mit dem move in einer Spalte keine 4 erreicht werden können
        heuristic_c = 0
        heuristic_d = 0
    elif streak_cd == 7:
        heuristic += 2
    else:                           # (streak_cd > 3) and (streak_cd < 7):
        heuristic += 1

    if streak_ef < 4:               # wenn mit dem move in einer rechts-Diagonalen keine 4 erreicht werden können
        heuristic_e = 0
        heuristic_f = 0
    elif streak_ef == 7:
        heuristic += 2
    else:
                                    # (streak_ef > 3) and (streak_ef < 7):
        heuristic += 1

    if streak_gh < 4:               # wenn mit dem move in einer links-Diagonalen keine 4 erreicht werden können
        heuristic_g = 0
        heuristic_h = 0
    elif streak_gh == 7:
        heuristic += 2
    else:
                                    # (streak_gh > 3) and (streak_gh < 7):
        heuristic += 1


    heuristic += heuristic_a + heuristic_b + heuristic_c + heuristic_d + heuristic_e + heuristic_f + heuristic_g + heuristic_h

    return heuristic


def generate_move_with_heuristic(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    # check which Player we are and safe the other
    if player == PLAYER1:
        other = PLAYER2
    else:
        other = PLAYER1

    # check if we have to block the other player so they can't win
    criticalcolumn = column_to_be_played_for_win(board, other)
    if criticalcolumn != -1:
        return criticalcolumn, saved_state

    possible_moves = []
    for i in range(7):
        if board[5, i] == NO_PLAYER:
            possible_moves.append(i)

    # possible_moves.astype(np.int8)
    print("possible moves: ", possible_moves)
    maxim, column_to_play = -1, -1
    for i in possible_moves:
        current = evaluate_heuristic(board, i, player)
        if current > maxim:
            maxim = current
            column_to_play = i

    action = column_to_play
    print("best heuristic: ", maxim, "column to be played: ", action)
    return action, saved_state


class MiniMax:
    def __init__(self, gameState: GameState, player: BoardPiece):
        gameStatesTree = []
        if player == 1:
            player = 2
        else:
            player = 1
        for a in range(7):
            gameStatesTree.append(GameState(player, str(a)))
            for b in range(7):
                gameStatesTree.append(GameState(player, str(a) + str(b)))
                for c in range(7):
                    gameStatesTree.append(GameState(player, str(a) + str(b) + str(c)))
                    for d in range(7):
                        gameStatesTree.append(GameState(player, str(a) + str(b) + str(c) + str(d), gameState))
        self.gameStatesTree = gameStatesTree


def applyMiniMax(gameStatesTree: np.array, gameState: GameState, depth: int, alpha: int, beta: int, maximinzingPlayer: bool):
    if depth == 0:
        print(depth)
        return gameState.computeScore()

    childrenOfCurrentGameState = findChildrenOfCurrentGameState(gameStatesTree, gameState.positionID)
    print(childrenOfCurrentGameState)

    if maximinzingPlayer:
        maxEval = -1000
        for child in childrenOfCurrentGameState:
            eval = applyMiniMax(gameStatesTree, child, depth-1, alpha, beta, False)
            child.optimalScore = eval
            maxEval = max(maxEval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval
    else:
        minEval = 1000
        for child in childrenOfCurrentGameState:
            eval = applyMiniMax(gameStatesTree, child, depth-1, alpha, beta, True)
            child.optimalScore = eval
            minEval = min(minEval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval


def findChildrenOfCurrentGameState(gameStatesTree: np.array, positionIDCurrentGameState: str) -> np.array:
    if len(positionIDCurrentGameState) == 4:
        return []

    childrenIndexes = np.empty([1, ], dtype=int)
    for idx, element in enumerate(gameStatesTree):
        if len(childrenIndexes) >= 7:
            break

        if element.positionID.startswith(positionIDCurrentGameState) and len(element.positionID) == len(positionIDCurrentGameState) + 1:
            childrenIndexes = np.append(childrenIndexes, idx)

    return np.take(gameStatesTree, childrenIndexes)


def generateMoveWithMiniMax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:

    gameState = GameState(player, "", board)
    miniMax = MiniMax(gameState, player)
    optimalScore = applyMiniMax(miniMax.gameStatesTree, gameState, 4, -1000, 1000, True)
    relevantGameStates = findChildrenOfCurrentGameState(miniMax.gameStatesTree, "")
    playerAction = int([gs for gs in relevantGameStates if gs.optimalScore == optimalScore][0].positionID)
    return playerAction, saved_state
