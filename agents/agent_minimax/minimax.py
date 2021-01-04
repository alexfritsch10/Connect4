import numpy as np

from agents.agent_minimax.heuristic import evaluate_heuristic
from agents.common import BoardPiece, PLAYER1, PLAYER2, PlayerAction, apply_player_action, connected_four, SavedState, \
    Optional, Tuple, NO_PLAYER, column_to_be_played_for_win, initialize_game_state, other_player, pretty_print_board, \
    move_is_possible
from agents.agent_minimax.gameState import GameState


def set_value_recursiv(node: GameState, depth: int, player: BoardPiece):
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
                set_value_recursiv(i, depth+1, player)
        if player == node.player:
            node.value = max(child.value for child in node.children if child.value is not None)
        elif player == other_player(node.player):
            node.value = min(child.value for child in node.children if child.value is not None)

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
    set_value_recursiv(root, 0, player)

    # printing subtree..
    # print('root: ', root.positionID, root.value)
    # print('children: ')
    # for i in root.children:
    #    print(' ', i.positionID, i.value)

    # calculate moves for root:
    moves = []
    possible_moves = []
    for i in root.children:
        if i.value is not None and i.value == root.value:
            moves.append(root.children.index(i))

    print("moves to choose from: ")
    print(moves)

    for j in moves:
        if move_is_possible(board, j):
            possible_moves.append(j)
    move = np.random.choice(possible_moves, 1)

    return move, None
