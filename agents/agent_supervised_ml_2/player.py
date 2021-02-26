import copy
import random
import numpy as np
from common import apply_player_action


class Player:

    def __init__(self, number, strategy='random', model=None):
        self.number = number
        self.strategy = strategy
        self.model = model


    def getMove(self, availableMoves: np.ndarray, board: np.ndarray) -> int:
        if self.strategy == "random":
            return availableMoves[random.randrange(0, len(availableMoves))]
        else:
            for availableMove in availableMoves:
                boardCopy = copy.deepcopy(board)
                apply_player_action(boardCopy, availableMove, self.number)
                winner = self.model.predict(boardCopy)
                if winner == self.number:
                    return availableMove
            return availableMoves[random.randrange(0, len(availableMoves))]

    def getNumber(self):
        return self.number
