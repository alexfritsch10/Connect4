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
            maxValue = 0
            bestMove = availableMoves[0]
            for availableMove in availableMoves:
                boardCopy = copy.deepcopy(board)
                apply_player_action(boardCopy, availableMove, self.number)
                if self.number == 1:
                    value = self.model.predict(boardCopy, 2)
                else:
                    value = self.model.predict(boardCopy, 0)
                if value > maxValue:
                    maxValue = value
                    bestMove = availableMove
            return bestMove

    def getNumber(self):
        return self.number
