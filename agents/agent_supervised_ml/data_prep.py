import numpy as np

from agents.common import initialize_game_state, apply_player_action, PLAYER1, other_player, pretty_print_board


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

    file = open('/Users/kira/Desktop/uni/Connect4/agents/agent_supervised_ml/scores2.txt', 'r')

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
    when there are more min scores this function picks one randomly so there are no duplicates
    :return:
    """

    file = open('/Users/kira/Desktop/uni/Connect4/agents/agent_supervised_ml/scores2.txt', 'r')

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

    X = np.append(X, y, axis=1)
    X = np.unique(X, axis=0)
    y = X[:, -1]
    X = X[:, :-1]

    return X, y
