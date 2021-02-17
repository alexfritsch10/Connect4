import numpy as np


def clean_scores() -> (np.array, np.array):

    scores = np.ndarray((0, 1), int)
    moves = np.ndarray((0, 42), int)
                                                                                                            # data:
    # file = open('/Users/kira/Desktop/uni/Connect4/agents/agent_supervised/position_ids_scores.txt', 'r')  # small
    file = open('/Users/kira/Desktop/uni/Connect4/agents/agent_supervised_ml/scores.txt', 'r')            # medium
    # file = open('/Users/kira/Desktop/uni/Connect4/agents/agent_supervised_ml/scores_large.txt', 'r')        # large
    for line in file:
        line = line.strip()             # go through file line by line
        if line:                                    # only non-empty lines
            moves_and_score = line.split(' ')                   # split string into board and score (label)
            move = list(moves_and_score[0])
            move = np.asarray(move, int)
            for i in range(42 - len(move)):      # 41-len(moves) 0en müssen hinzugefügt werden
                move = np.append(move, 0)
            moves = np.vstack([moves, move])
            score = np.array([int(moves_and_score[1])])
            scores = np.vstack([scores, score])

    file.close()
    # print('moves: ')
    # print(moves)
    # print('scores: ')
    # print(scores)

    return moves, scores
