import numpy as np
from agents.common import BoardPiece, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, SavedState, \
    Optional, Tuple, NO_PLAYER, column_to_be_played_for_win, initialize_game_state, other_player, pretty_print_board, \
    move_is_possible
from agents.agent_minimax.gameState import GameState

"""
class MiniMax:
    def __init__(self, currentBoard: np.array, player: BoardPiece):

        # ich hab das mal um den if-else teil gekürzt, dafür habe ich eine funktion in common geschrieben die einfach den
        # anderen player zurückgibt, dann kann egal sein wer gerade tatsächlich spielt: player und other_player(player)

        gameStatesTree = []
        gameStatesTree.append(GameState(player, "", currentBoard))
        for a in range(7):
            gameStatesTree.append(GameState(other_player(player), str(a)))
            for b in range(7):
                gameStatesTree.append(GameState(player, str(a) + str(b)))
                for c in range(7):
                    gameStatesTree.append(GameState(other_player(player), str(a) + str(b) + str(c)))
                    for d in range(7):
                        gameStatesTree.append(GameState(player, str(a) + str(b) + str(c) + str(d), currentBoard))

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

    miniMax = MiniMax(board, player)
    optimalScore = applyMiniMax(miniMax.gameStatesTree, miniMax.gameStatesTree[0], 4, -1000, 1000, True)

    relevantGameStates = findChildrenOfCurrentGameState(miniMax.gameStatesTree, "")

    playerAction = int([gs for gs in relevantGameStates if gs.optimalScore == optimalScore][0].positionID)

    return playerAction, saved_state
"""


def set_value_recursiv(node: GameState, depth: int):
    """
    computes values of nodes in the tree recursively. If we're 4 moves in future, build the board for this node
    and set it's value using the heuristic. For all other nodes set value to the best value of child nodes
    """
    # print('nodeID, depth:', node.positionID, depth)
    if depth == 4:
        node.buildGameStateFromID()
        node.set_value()
        return
    else:
        for i in node.children:
            if i.value is None:
                set_value_recursiv(i, depth+1)
        node.value = max(- child.value for child in node.children if child.value is not None)

        # node.bmove = np.argmax(- child.value for child in node.children)
        return


def generateMoveWithMiniMax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    chooses move with minimax tree.
    First build tree recursively, calling GameState Class with root and given board.
    Then set values for all nodes in the tree, so we know which node is best for us
    and see what moves there are.
    If there are equally optimal moves pick one randomly.
    """

    # print('board: ', pretty_print_board(board))
    root = GameState(player, '', board)
    # print('tree built')
    set_value_recursiv(root, 0)

    # calculate moves for root:
    moves = []
    possible_moves = []
    for i in root.children:
        if i.value is not None and - i.value == root.value:
            moves.append(root.children.index(i))

    for j in moves:
        if move_is_possible(board, j):
            possible_moves.append(j)
    move = np.random.choice(possible_moves, 1)

    return move, None
