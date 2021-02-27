import numpy as np
from sklearn.model_selection import train_test_split

from agents.agent_supervised_ml.classification import linear_svm, k_nearest_neighbours, decision_tree, \
    logistic_regression, naive_bayes, multilayer_perceptron, information_on_split_data_v1, \
    information_on_split_data_v2
from agents.agent_supervised_ml.data_prep import clean_scores_version1, compute_moves_v2, create_dataset, \
    compute_moves_v2_without_duplicates, eliminate_duplicates
from agents.agent_supervised_ml.pytorch_linreg import linear_regression
from agents.agent_supervised_ml.pytorch_logisticreg import logistic_regression_pytorch
from agents.agent_supervised_ml.supervised import board_to_move_seq
from agents.common import PLAYER2, BoardPiece, PLAYER1, pretty_print_board, apply_player_action, other_player
import matplotlib.pyplot as plt

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
        a, b = clean_scores_version1()

        assert isinstance(a, np.ndarray)
        assert a.shape[1] == 41
        assert isinstance(b, np.ndarray)

    def test_compute_moves(self):
        compute_moves_v2()

    def test_compute_moves_without_duplicates(self):
        compute_moves_v2_without_duplicates()

    def test_create_dataset(self):
        create_dataset()

    def test_eliminate_duplicates(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)
        print('X: ', X, 'y: ', y)

        assert X.shape[0] == y.shape[0]

    def test_information_on_split_data_v1(self):
        information_on_split_data_v1()

    def test_information_on_split_data_v2(self):
        information_on_split_data_v2()

    def test_linear_svm(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = linear_svm(X_train, X_test, y_train, y_test)

    def test_k_nearest_neighbours(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = k_nearest_neighbours(X_train, X_test, y_train, y_test)

    def test_decision_tree(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = decision_tree(X_train, X_test, y_train, y_test)

    def test_logistic_regression(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = logistic_regression(X_train, X_test, y_train, y_test)

    def test_naive_bayes(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = naive_bayes(X_train, X_test, y_train, y_test)


    def test_multilayer_perceptron(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        acc = multilayer_perceptron(X_train, X_test, y_train, y_test)

    def test_accuracies(self):
        accuracies = []
        classifiers = ['lin_svm', 'k_n_neighb', 'dec_tree', 'log_regr', 'naive_bayes', 'mlp']
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)

        # split training and test data (test size 20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        acc = linear_svm(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        acc = k_nearest_neighbours(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        acc = decision_tree(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        acc = logistic_regression(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        acc = naive_bayes(X_train, X_test, y_train, y_test)
        accuracies.append(acc)
        acc = multilayer_perceptron(X_train, X_test, y_train, y_test)
        accuracies.append(acc)

        plt.bar(classifiers, accuracies)
        # Namimg the x and y axis
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracies')
        # Giving the tilte for the plot
        plt.title('Accuracies of our Classifiers')
        # Saving the plot as a 'png'
        # plt.savefig('ACCPlot.png')
        plt.show()

        print(accuracies)
        print(classifiers)

        assert True

