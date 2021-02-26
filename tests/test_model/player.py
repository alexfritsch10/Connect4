import random
import numpy as np
from agents.agent_supervised_ml.supervised import generate_move_supervised


class Player:

    def __init__(self, number, strategy):
        self.number = number
        self.strategy = strategy

    def getMove(self, availableMoves: np.ndarray, board: np.ndarray) -> int:
        if self.strategy == "random":
            return availableMoves[random.randrange(0, len(availableMoves))]
        else:
            return generate_move_supervised(board, self.number, None)[0]

    def getNumber(self):
        return self.number
