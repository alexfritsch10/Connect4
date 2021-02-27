import numpy as np
import random

from agents.agent_supervised_ml.data_prep import move_seq_to_board_input_vector
from agents.common import BoardPiece, apply_player_action, connected_four, other_player, \
 initialize_game_state, board_to_lists


def generate_random_position_ids(n: int):
    """
    Computes random position Ids and writes them into a local file

    Parameters
    ----------
    n : int
        Number of position Ids that should be generated
    """
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


def clean_scores_version1() -> (np.array, np.array):
    """
    Reads in txt file to generate a clean dataset

    Returns
    -------
    Cleaned X and y
    """

    scores = np.ndarray((0, 1), int)
    moves = np.ndarray((0, 42), int)

    file = open('agents/agent_supervised_ml/position_ids_scores.txt', 'r')

    for line in file:
        line = line.strip()
        # only processes non empty lines
        if line:
            moves_and_score = line.split(' ')
            move = list(moves_and_score[0])
            move = np.asarray(move, int)
            for i in range(42 - len(move)):
                move = np.append(move, 0)
            moves = np.vstack([moves, move])
            score = np.array([int(moves_and_score[1])])
            scores = np.vstack([scores, score])

    file.close()

    moves = move_seq_to_board_input_vector(moves)

    return moves, scores


def create_dataset():
    """
    creates .txt file with dataset, recycling of already generated data, for structured relabeling,
    so it's possible to find the best moves
    """

    x_old, y_old = clean_scores_version1()

    # delete duplicates
    x_old = np.unique(x_old, axis=0)

    file = open('agents/agent_supervised_ml/unlabeled2.txt', "a")

    for row in x_old:
        string = ''
        move_seq = row[row != 0]
        for move in move_seq:
            string = string + str(move)
        for i in range(1, 8):
            file.write(string + str(i) + '\n')

    file.close()
