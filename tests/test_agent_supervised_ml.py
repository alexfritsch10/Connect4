import numpy as np
from sklearn.model_selection import train_test_split

from agents.agent_supervised_ml.classification import linear_svm, k_nearest_neighbours, decision_tree, \
    logistic_regression, naive_bayes, multilayer_perceptron
from agents.agent_supervised_ml.data_prep import compute_moves_v2, \
    compute_moves_v2_without_duplicates, eliminate_duplicates, information_on_split_data_v2, \
    move_seq_to_board_input_vector
from agents.agent_supervised_ml.generate_data_set import clean_scores_version1, create_dataset, \
    generate_random_position_ids
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

    # TESTS FOR generate_data_set.py:

    def test_generate_random_position_ids(self):
        n = 1
        generate_random_position_ids(n)

    def test_clean_scores_version1(self):
        a, b = clean_scores_version1()

        assert isinstance(a, np.ndarray)
        assert a.shape[1] == 41
        assert isinstance(b, np.ndarray)

    def test_create_dataset(self):
        create_dataset()


    # TESTS FOR data_prep.py:

    def test_move_seq_to_board_input_vector(self):

        board_ex = np.zeros((6, 7), BoardPiece)
        board_ex[0, 0] = PLAYER1
        board_ex[0, 3] = PLAYER2
        board_ex[0, 1] = PLAYER1
        board_ex[1, 3] = PLAYER2
        board_ex[0, 5] = PLAYER1
        board_ex[0, 6] = PLAYER2

        move_sequences = np.ndarray((0, 42), int)
        move_sequence = np.array([1, 4, 2, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        move_sequences = np.vstack([move_sequences, move_sequence])
        boards = move_seq_to_board_input_vector(move_sequences)

        assert isinstance(boards, np.ndarray)

    def test_compute_moves_v2(self):
        compute_moves_v2()

    def test_compute_moves_v2_without_duplicates(self):
        compute_moves_v2_without_duplicates()

    def test_eliminate_duplicates(self):
        X, y = compute_moves_v2_without_duplicates()
        X, y = eliminate_duplicates(X, y)
        print('X: ', X, 'y: ', y)

        assert X.shape[0] == y.shape[0]

    def test_information_on_split_data_v2(self):
        information_on_split_data_v2()


    # TESTS FOR classification.py:

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

