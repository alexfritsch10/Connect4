import numpy as np

from agents.agent_supervised_ml.classification import linear_svm, k_nearest_neighbours, decision_tree, \
    logistic_regression, naive_bayes, multilayer_perceptron, information_on_split_data
from agents.agent_supervised_ml.data_prep import clean_scores
from agents.agent_supervised_ml.pytorch_linreg import linear_regression
from agents.agent_supervised_ml.pytorch_logisticreg import logistic_regression_pytorch
from agents.agent_supervised_ml.supervised import board_to_move_seq
from agents.common import PLAYER2, BoardPiece, PLAYER1, pretty_print_board, apply_player_action, other_player

board2 = np.zeros((6, 7), BoardPiece)  # possible played board
board2[0, 1] = PLAYER2
board2[0, 2] = PLAYER2
board2[0, 3] = PLAYER1
board2[0, 4] = PLAYER1
board2[1, 1] = PLAYER2
board2[1, 2] = PLAYER1
board2[1, 3] = PLAYER2
board2[1, 4] = PLAYER2
board2[2, 2] = PLAYER2
board2[2, 3] = PLAYER1
board2[2, 4] = PLAYER1
board2[3, 2] = PLAYER1
board2[3, 3] = PLAYER1

class Tests:

    def test_board_to_move_seq(self):
        print(pretty_print_board(board2))
        move_seq = board_to_move_seq(board2)
        print(move_seq)

        board = np.zeros((6, 7), BoardPiece)
        player = PLAYER1
        for i in move_seq:
            apply_player_action(board, i-1, player)
            player = other_player(player)

        print(pretty_print_board(board))

        assert board.all(board2)


    def test_linear_regression(self):
        linear_regression()

    def test_logistic_regression_pytorch(self):
        logistic_regression_pytorch()

    def test_clean_scores(self):
        a, b = clean_scores()

        assert isinstance(a, np.ndarray)
        assert a.shape[1] == 41
        assert isinstance(b, np.ndarray)

    def test_information_on_split_data(self):
        information_on_split_data()

    def test_linear_svm(self):
        linear_svm()

    def test_k_nearest_neighbours(self):
        k_nearest_neighbours()

    def test_decision_tree(self):
        decision_tree()

    def test_logistic_regression(self):
        logistic_regression()

    def test_naive_bayes(self):
        naive_bayes()

    def test_multilayer_perceptron(self):
        multilayer_perceptron()
