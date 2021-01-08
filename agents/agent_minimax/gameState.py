import numpy as np

from agents.agent_minimax.heuristic import evaluate_heuristic
from agents.common import BoardPiece, apply_player_action, connected_four, column_to_be_played_for_win, other_player, \
    move_is_possible, pretty_print_board


class GameState:
    def __init__(self, player: BoardPiece, positionID: str, board: np.ndarray):   # vorher: = None
        """
        recursively building game tree, every middle node (those who are less than depth 4) has list of children,
        building when GameState with root is called. All nodes on depth 4 get a board copy, it'll get modified later by
        buildGameStateFromID and the score of all nodes is None at the beginning.
        """
        self.positionID = positionID
        self.player = player
        self.status = None                 # added self.status in case a combination of moves is impossible
        self.score = None
        if len(positionID) < 4:
            # print('positionID <4:', positionID)
            self.children = []
            for i in range(7):
                self.children.append(GameState(other_player(player), positionID + str(i), board))
        else:
            # print('positionID >= 4: ', positionID)
            self.board = board.copy()

    def compute_score(self):
        """
        uses heuristic to set score of node,
        for loop to find the best column according to heuristic makes it very slow 
        """
        self.score = -1000
        for i in range(7):
            if move_is_possible(self.board, i):
                current = evaluate_heuristic(self.board, i, self.player)
                if current > self.score:
                    self.score = current

        #print('Set score ' + str(self.score) + ' for node ' + self.positionID + ' with board ' + pretty_print_board(self.board))

        if self.score == -1000:
            self.status = 'impossible'
            print('no possible moves for Node:')
            print(self.positionID)


    def buildGameStateFromID(self):
        """
        plays 4 moves on self.board if they're possible
        """
        # positionID = eg 4256 -> 1. Move: 4th Column, 2. Move: 2nd column
        player = self.player
        for col in self.positionID:
            if move_is_possible(self.board, int(col)):                            # only apply possible moves
                apply_player_action(self.board, int(col), player)
                player = other_player(player)
            else:
                self.status = 'impossible'
                break
        # print(pretty_print_board(self.board))
        # hier hast du den player am anfang auf 1 gesetzt, egal ob player 1 oder player 2 spielt





