import numpy as np

from agents.agent_minimax.heuristic import evaluate_heuristic
from agents.common import BoardPiece, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, SavedState, \
    Optional, Tuple, NO_PLAYER, column_to_be_played_for_win, initialize_game_state, other_player, pretty_print_board, \
    move_is_possible
from agents.agent_minimax.gameState import GameState


def minimax(node: GameState, depth: int, alpha: int, beta: int, maximizingPlayer: bool):
    """
    computes scores of nodes in the tree recursively. If we're 4 moves in future, build the board for this node
    and set it's score using the heuristic. For all other nodes set score to the best score of child nodes
    """
    #print('Current depth: ' + str(depth) + ' for maximizingPlayer ? ' + str(maximizingPlayer))
    if depth == 0:
        node.buildGameStateFromID()
        if node.status != 'impossible':
            node.compute_score()
        return node.score

    #print('Maximizing Player: ' + str(maximizingPlayer))

    if maximizingPlayer:
        node.score = -1000
        for child in node.children:
            curVal = minimax(child, depth-1, alpha, beta, False)
            #print('(MAX) Set node score of node ' + str(node.positionID) + ' from ' + str(node.score) + ' to ' + str(max(node.score, curVal)))
            node.score = max(node.score, curVal)
            alpha = max(alpha, curVal)
            if beta <= alpha:
                break
        return node.score

    else:
        node.score = 1000
        for child in node.children:
            curVal = minimax(child, depth-1, alpha, beta, True)
            #print('(MIN) Set node score of node ' + str(node.positionID) + ' from ' + str(node.score) + ' to ' + str(min(node.score, curVal)))
            node.score = min(node.score, curVal)
            beta = min(beta, curVal)
            if beta <= alpha:
                break

        return node.score


def generateMoveWithMiniMax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    chooses move with minimax tree.
    First build tree recursively, calling GameState Class with root and given board.
    Then set scores for all nodes in the tree, so we know which node is best for us
    and see what moves there are.
    If there are equally optimal moves pick one randomly.
    """

    # print('board: ', pretty_print_board(board))
    root = GameState(player, '', board)
    # print('tree built')
    minimax(root, 4, -np.inf, np.inf, True)

    # printing subtree..
    # print('root: ', root.positionID, root.score)
    # print('children: ')
    # for i in root.children:
    #    print(' ', i.positionID, i.score)

    # calculate moves for root:
    moves = []
    possible_moves = []
    for i in root.children:
        if i.score == root.score:
            print('Add move for col ' + str(root.children.index(i)) + ' with score ' + str(i.score))
            moves.append(root.children.index(i))

    print("moves to choose from: ")
    print(moves)
    for j in moves:
        if move_is_possible(board, j):
            possible_moves.append(j)
    move = np.random.choice(possible_moves, 1)
    print(possible_moves)

    return move, None
