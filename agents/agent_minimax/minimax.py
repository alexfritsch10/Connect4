import numpy as np

from agents.common import BoardPiece, PlayerAction, SavedState, Optional, Tuple, move_is_possible
from agents.agent_minimax.gameState import GameState


def minimax(node: GameState, depth: int, alpha: int, beta: int, maximizingPlayer: bool):
    """
    computes scores of nodes in the tree recursively. If we're 4 moves in future, build the board for this node
    and set it's score using the heuristic. For all other nodes set score to the best score of child nodes

    Parameters
    ----------
    node : GameState
           Current node in the Tree
    depth: int
           Current depth in the Tree
    alpha: int
           Highest score achieved by the current subtree
    beta : int
           Lowest score achieved by the current subtree
    maximizingPlayer: bool
           Current player Strategy
    """

    # static evaluation of a GameState if at the bottom of the tree
    if depth == 0:
        node.buildGameStateFromID()
        if node.possible:
            node.compute_score()
        return node.score

    # recursively computing the best move
    if maximizingPlayer:
        node.score = -1000
        for child in node.children:
            curVal = minimax(child, depth-1, alpha, beta, False)
            node.score = max(node.score, curVal)
            alpha = max(alpha, curVal)
            if beta <= alpha:
                break

        return node.score

    else:
        node.score = 1000
        for child in node.children:
            curVal = minimax(child, depth-1, alpha, beta, True)
            node.score = min(node.score, curVal)
            beta = min(beta, curVal)
            if beta <= alpha:
                break

        return node.score


def generateMoveWithMiniMax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Provides interface to make Minimax Agent able to play
    Then set scores for all nodes in the tree, so we know which node is best for us
    and see what moves there are.
    If there are equally optimal moves pick one randomly.

    Parameters
    ----------
    board : np.ndarray
            Root GameState to build tree
    player: BoardPiece
            Player to find optimal moves for
    saved_state: Optional[SavedState]
            Debugging param

    Returns
    -------
    Optimal Move for the current board and player, and saved_state

    """

    # creating a minimax tree with depth 4
    root = GameState(player, '', board)
    minimax(root, 4, -np.inf, np.inf, True)

    # calculate moves for root
    moves = []
    possible_moves = []
    for i in root.children:
        if i.score == root.score:
            moves.append(root.children.index(i))

    # only keeps possible moves from the calculated ones
    for j in moves:
        if move_is_possible(board, j):
            possible_moves.append(j)
    print("Moves to choose from: " + str(possible_moves))
    move = np.random.choice(possible_moves, 1)

    return move, None
