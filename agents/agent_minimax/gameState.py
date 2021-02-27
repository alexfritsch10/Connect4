import numpy as np

from agents.common import BoardPiece, apply_player_action, other_player, move_is_possible, PlayerAction
from agents.agent_minimax.heuristic import evaluate_heuristic

class GameState:
    def __init__(self, player: BoardPiece, positionID: str, board: np.ndarray):
        """
        recursively building game tree, every middle node (those who are less than depth 4) has list of children,
        building when GameState with root is called. All nodes on depth 4 get a board copy, it'll get modified later by
        buildGameStateFromID and the score of all nodes is None at the beginning.
        """
        self.positionID = positionID
        self.player = player
        self.possible = True
        self.score = 0
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
        uses heuristic to set score of node
        """
        action = PlayerAction(self.positionID[3])
        self.score = evaluate_heuristic(self.board, action, self.player)

    def buildGameStateFromID(self):
        """
        plays 4 moves on self.board if they're possible
        """
        # positionID = eg 4256 -> 1. Move: 4th Column, 2. Move: 2nd column
        player = self.player
        for col in self.positionID:
            action = PlayerAction(int(col))
            if move_is_possible(self.board, action):
                apply_player_action(self.board, action, player)
                player = other_player(player)
            else:
                self.status = False
                break
