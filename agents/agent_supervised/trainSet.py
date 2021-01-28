import numpy as np
import random


def generateRandomPositionIDs(n):

    possibleMovesInitial = np.array([])
    for i in range(7):
        possibleMovesInitial = np.append([str(i)] * 6, possibleMovesInitial)

    posIdLengths = np.random.randint(4, 43, size=n)
    positionIds = np.array([])
    for posIdLength in posIdLengths:
        possibleMoves = possibleMovesInitial
        posId = ""
        for row in range(posIdLength):
            move = random.choice(possibleMoves)
            posId += move
            moveIndex = np.where(possibleMoves == move)[0][0]
            possibleMoves = np.delete(possibleMoves, moveIndex)

        positionIds = np.append(posId, positionIds)

    return positionIds


