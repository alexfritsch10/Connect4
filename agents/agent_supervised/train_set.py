import numpy as np
import random
from agents.common import BoardPiece, apply_player_action, connected_four, other_player, \
 initialize_game_state, board_to_lists


def generate_random_position_ids(n):
    possibleMovesInitial = np.array([])
    for i in range(7):
        possibleMovesInitial = np.append([str(i+1)] * 6, possibleMovesInitial)

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
    print(positionIds)

    f = open('agents/agent_supervised/position_ids.txt', "a")
    for positionId in positionIds:
        f.write(positionId + "\n")
    f.close()


def generate_train_set() -> (np.ndarray, np.ndarray):
    f = open('agents/agent_supervised/position_ids_scores.txt', 'r')
    lines = f.readlines()

    inputs = np.ndarray([])
    labels = np.zeros((6, 7, 0))

    print(inputs)


    for idx, line in enumerate(lines):
        if idx > 20:
            break
        line = line.strip()
        if line != "":
            lineSplit = line.split(" ")

            positionId = lineSplit[0]
            print(positionId)
            shiftedPositionId = shiftBackPositionId(positionId)
            print(shiftedPositionId)
            inputElement = buildGameStateFromID(shiftedPositionId)
            print(inputElement)
            if idx == 0:
                inputs = inputElement
            else:
                inputs = np.append(inputs, inputElement, axis=1)
            print("---")
            print(inputs)
            print("-|-")
            print(inputs.shape())

            labelElement = lineSplit[1]
            labels = np.append(labelElement, labels)

    f.close()
    return inputs, labels


def shiftBackPositionId(positionId: str) -> str:
    shiftedPositionId = ""
    for move in positionId:
        shiftedPositionId += str(int(move) - 1)
    return shiftedPositionId


def buildGameStateFromID(positionId: str) -> np.ndarray:
    player = 1
    board = initialize_game_state()
    for col in positionId:
        apply_player_action(board, int(col), player)
        player = other_player(player)
    return board
