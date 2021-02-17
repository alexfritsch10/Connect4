import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from agents.agent_supervised_ml.data_prep import clean_scores
from agents.common import PlayerAction

"""
labels=[-18, -17, -16, 15, -14, -13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -15, 16, 17, 18]
"""


def information_on_split_data():

    X, y = clean_scores()
    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print('(samples, number of features): ', X_train.shape)

    unique = np.unique(X_train, axis=0)
    print('unique shape: ', unique.shape)

    count_0 = np.count_nonzero(y_train == 0)
    count_n10 = np.count_nonzero(y_train == -10)
    count_10 = np.count_nonzero(y_train == 10)
    count_n18 = np.count_nonzero(y_train == -18)
    count_18 = np.count_nonzero(y_train == 18)
    count_n5 = np.count_nonzero(y_train == -5)
    count_5 = np.count_nonzero(y_train == 5)
    count_n6 = np.count_nonzero(y_train == -6)
    count_6 = np.count_nonzero(y_train == 6)
    count_n7 = np.count_nonzero(y_train == -7)
    count_7 = np.count_nonzero(y_train == 7)
    count_n8 = np.count_nonzero(y_train == -8)
    count_8 = np.count_nonzero(y_train == 8)
    count_n9 = np.count_nonzero(y_train == -9)
    count_9 = np.count_nonzero(y_train == 9)
    count_n11 = np.count_nonzero(y_train == -11)
    count_11 = np.count_nonzero(y_train == 11)
    count_n12 = np.count_nonzero(y_train == -12)
    count_12 = np.count_nonzero(y_train == 12)
    count_n13 = np.count_nonzero(y_train == -13)
    count_13 = np.count_nonzero(y_train == 13)
    count_n14 = np.count_nonzero(y_train == -14)
    count_14 = np.count_nonzero(y_train == 14)
    count_n15 = np.count_nonzero(y_train == -15)
    count_15 = np.count_nonzero(y_train == 15)
    count_n16 = np.count_nonzero(y_train == -16)
    count_16 = np.count_nonzero(y_train == 16)
    count_n17 = np.count_nonzero(y_train == -17)
    count_17 = np.count_nonzero(y_train == 17)


    print(count_0, count_n5, count_5, count_n6, count_6, count_n7, count_7, count_n8, count_8, count_n9, count_9,
          count_n10, count_10, count_n11, count_11, count_n12, count_12, count_n13, count_13, count_n14, count_14,
          count_n15, count_15, count_n16, count_16, count_n17, count_17, count_n18, count_18)


def linear_svm():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf_svm = svm.SVC()

    clf_svm.fit(X_train, y_train)

    y_pred = clf_svm.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)


def k_nearest_neighbours():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf = KNeighborsClassifier(n_neighbors=15)

    clf.fit(X_train, y_train)

    print(X_test)
    print(X_test[1])

    y_pred_single = clf.predict([X_test[1]])
    print(y_pred_single)
    y_pred = clf.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)


def decision_tree():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf_dec = DecisionTreeClassifier()

    clf_dec.fit(X_train, y_train)

    y_pred = clf_dec.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)


def logistic_regression():

    X, y = clean_scores()
    # n_samples, n_features = X.shape
    # print('move seq: ', move_seq)
    # move_seq = move_seq.reshape(1, -1)

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf_log = LogisticRegression()

    clf_log.fit(X_train, y_train)

    y_pred = clf_log.predict(X_test)

    # print('y_pred: ', y_pred)
    # print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    # print('f1 score: ', f1_sc)

    #score = clf_log.predict(move_seq)
    #print('score: ', score)
    #return score


def naive_bayes():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf_gnb = GaussianNB()

    clf_gnb.fit(X_train, y_train)

    y_pred = clf_gnb.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)


def multilayer_perceptron():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # clf_svm = svm.SVC(kernel='linear')
    clf_MLP = MLPClassifier(hidden_layer_sizes=(160, 10), activation='logistic', random_state=1, max_iter=210)
    # William and Nico Parameter:
    #clf_MLP = MLPClassifier(hidden_layer_sizes=(126*5), activation='logistic', random_state=1, max_iter=1500,
    #                        alpha=0.001, n_iter_no_change=10, learning_rate='adaptive', learning_rate_init=0.005,
    #                        shuffle=True, tol=0.0001, verbose=0)

    clf_MLP.fit(X_train, y_train)

    y_pred = clf_MLP.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    # y_test = y_test.flatten()
    # print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)
