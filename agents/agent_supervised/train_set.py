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


def generate_train_set():
    f = open('agents/agent_supervised/position_ids_scores.txt', 'r')
    lines = f.readlines()
    trainSet = open('agents/agent_supervised/data_set.txt', 'a')
    trainSet.write("player, doubleMill, number3Connect, number2Connect\n")

    for line in lines:
        line = line.strip()
        if line != "":
            lineSplit = line.split(" ")
            positionId = lineSplit[0]
            score = lineSplit[1]
            shiftedPositionId = shiftBackPositionId(positionId)
            features = compute_features_for_position_id(shiftedPositionId)
            trainSet.write(features + score + "\n")

    f.close()
    trainSet.close()


def shiftBackPositionId(positionId: str) -> str:
    shiftedPositionId = ""
    for move in positionId:
        shiftedPositionId += str(int(move) - 1)
    return shiftedPositionId


def buildGameStateFromID(positionId: str) -> (int, np.ndarray):
    player = 1
    board = initialize_game_state()
    for col in positionId:
        apply_player_action(board, int(col), player)
        player = other_player(player)
    return player, board


def compute_features_for_position_id(positionId: str) -> str:
    player, board = buildGameStateFromID(positionId)

    features = str(player)
    features += ","

    features += str(checkForDoubleMill(board, player))
    features += ","

    features += str(numberOfStreaksWithLength(board, player, 3))
    features += ","

    features += str(numberOfStreaksWithLength(board, player, 2))
    features += ","

    return features


def checkForDoubleMill(board: np.ndarray, player: BoardPiece) -> bool:
    moves = np.array([])
    for i in range(7):
        board_cpy = board.copy()
        if connected_four(apply_player_action(board_cpy, i, player), player):
            moves = np.append(i, moves)

    return len(moves) > 1


def numberOfStreaksWithLength(board: np.ndarray, player: BoardPiece, requiredStreakLength: int) -> int:
    lists = board_to_lists(board)
    numberOfStreaks = 0
    streakLength = 0
    for rowList in lists:
        for col in rowList:
            if col == player:
                streakLength += 1
                if streakLength > (requiredStreakLength - 1):
                    numberOfStreaks += 1
            else:
                streakLength = 0
        streakLength = 0

    return numberOfStreaks
