import numpy as np
from agents.common import initialize_game_state, apply_player_action, PLAYER1, other_player, pretty_print_board
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def move_seq_to_board_input_vector(move_sequences: np.array) -> np.array:
    """
    takes X-Matrix with move sequences and convert each sequence into the board.
    """

    boards = np.ndarray((0, 42), int)
    for move_seq in move_sequences:
        player = PLAYER1
        board = initialize_game_state()
        for col in move_seq:
            apply_player_action(board, col - 1, player)
            player = other_player(player)

        boards = np.vstack([boards, board.flatten()])

    return boards


def compute_moves_v2() -> (np.array, np.array):
    """
    checks every 7 lines which moves got the best scores for a given board, creates X-Matrix and y-vector from .txt file
    :return:
    """

    file = open('agents/agent_supervised_ml/scores2.txt', 'r')

    count = 0
    move_scores = np.empty((7, 1))
    moves = np.ndarray((0, 1), int)         # labels
    boards = np.ndarray((0, 42), int)        # board representations as move sequences

    for line in file:
        if line == '\n':
            # invalid move
            count += 1
        else:
            board_and_move = line.split(' ')  # split string into board and score (label)
            board = board_and_move[0]
            board = list(board[:-1])
            board = np.asarray(board, int)
            for i in range(42 - len(board)):  # 41-len(moves) 0en müssen hinzugefügt werden
                board = np.append(board, 0)

            move_scores[count % 7] = int(board_and_move[1])
            count += 1

        if count % 7 == 0:
            if move_scores.size:                                                  # if move_scores isn't empty
                min = np.where(move_scores == np.amin(move_scores))
                for i in min[0]:
                    boards = np.vstack([boards, board])
                    move = np.array(i + 1)
                    moves = np.vstack([moves, move])

                move_scores = np.empty((7, 1))

    file.close()

    # print('X shape: ', boards.shape, 'y shape: ', moves.shape)
    # print('X: ', boards, 'y: ', moves)
    # print(boards[0])

    boards = move_seq_to_board_input_vector(boards)
    # print('X shape: ', boards.shape, 'y shape: ', moves.shape)
    # print('X: ', boards)
    # print(boards[0])

    return boards, moves


def compute_moves_v2_without_duplicates() -> (np.array, np.array):
    """
    checks every 7 lines which moves got the best scores for a given board, creates X-Matrix and y-vector from .txt file
    when there are more min scores this function picks one randomly so there are no ambiguous duplicates.
    :return:
    """

    file = open('agents/agent_supervised_ml/scores2.txt', 'r')

    count = 0
    move_scores = np.empty((7, 1))
    moves = np.ndarray((0, 1), int)         # labels
    boards = np.ndarray((0, 42), int)        # board representations as move sequences

    for line in file:
        if line == '\n':
            # invalid move
            count += 1
        else:
            board_and_move = line.split(' ')  # split string into board and score (label)
            board = board_and_move[0]
            board = list(board[:-1])
            board = np.asarray(board, int)
            for i in range(42 - len(board)):  # 42-len(moves) 0en müssen hinzugefügt werden
                board = np.append(board, 0)

            move_scores[count % 7] = int(board_and_move[1])
            count += 1

        if count % 7 == 0:
            if move_scores.size:                                                  # if move_scores isn't empty
                min = np.where(move_scores == np.amin(move_scores))
                move = np.array(int(np.random.choice(min[0])) + 1)
                boards = np.vstack([boards, board])
                moves = np.vstack([moves, move])

                # extra_features = np.vstack([extra_features, extra_feature])

                move_scores = np.empty((7, 1))

    file.close()

    boards = move_seq_to_board_input_vector(boards)
    print('X shape: ', boards.shape, 'y shape: ', moves.shape)  # , 'extra feature shape: ', extra_features.shape)
    print('X.unique shape: ', np.unique(boards, axis=0).shape)
    print('X: ', boards, 'y: ', moves)  # , 'extra features: ', extra_features)

    return boards, moves


def eliminate_duplicates(X: np.array, y: np.array) -> (np.array, np.array):
    """
    deletes duplicates in a given X-Matrix with y-vector
    :param X:
    :param y:
    :return:
    """

    X = np.append(X, y, axis=1)
    X = np.unique(X, axis=0)
    y = X[:, -1]
    X = X[:, :-1]

    return X, y


def information_on_split_data_v2():
    """
    visualizes different aspects about a data set, such as number of samples, number of features, or counting the
    occurrences of different outcomes
    :return:
    """
    # X, y = compute_moves()
    X, y = compute_moves_v2_without_duplicates()
    X, y = eliminate_duplicates(X, y)
    print(X.shape, y.shape)
    # split training and test data (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print('(samples, number of features): ', X_train.shape)

    unique = np.unique(X_train, axis=0)
    print('unique shape: ', unique.shape)

    count_1 = np.count_nonzero(y_train == 1)
    count_2 = np.count_nonzero(y_train == 2)
    count_3 = np.count_nonzero(y_train == 3)
    count_4 = np.count_nonzero(y_train == 4)
    count_5 = np.count_nonzero(y_train == 5)
    count_6 = np.count_nonzero(y_train == 6)
    count_7 = np.count_nonzero(y_train == 7)

    counts = [count_1, count_2, count_3, count_4, count_5, count_6, count_7]
    moves = ['1', '2', '3', '4', '5', '6', '7']

    plt.bar(moves, counts)
    # Namimg the x and y axis
    plt.xlabel('Moves')
    plt.ylabel('Counts')
    # Giving the tilte for the plot
    plt.title('Data Distribution')
    # Saving the plot as a 'png'
    # plt.savefig('DataDistributionPlot.png')
    plt.show()
