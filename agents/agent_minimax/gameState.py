import numpy as np
from common import BoardPiece, apply_player_action, connected_four, column_to_be_played_for_win


class GameState:
    def __init__(self, player: BoardPiece, positionID: str, board: np.ndarray = None):
        self.positionID = positionID
        self.player = player
        if board is not None:
            self.gameState = board

    def computeScore(self):
        if connected_four(self.gameState, self.player):
            return 100
        elif column_to_be_played_for_win(self.gameState, self.player) != -1:
            return 50
        else:
            return 0

    def buildGameStateFromID(self):
        # positionID = eg 4256 -> 1. Move: 4th Column, 2. Move: 2nd column
        player = 1
        for col in self.positionID:
            apply_player_action(self.gameState, int(col), player)
            if player == 1:
                player = 2
            else:
                player = 1





