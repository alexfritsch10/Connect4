import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def linear_svm(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    # clf_svm = svm.SVC(kernel='linear')
    clf_svm = svm.SVC()

    clf_svm.fit(X_train, y_train)

    y_pred = clf_svm.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    return acc


def k_nearest_neighbours(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    clf = KNeighborsClassifier(n_neighbors=15)

    clf.fit(X_train, y_train)

    print(X_test)
    print(X_test[1])

    y_pred_single = clf.predict([X_test[1]])
    print(y_pred_single)
    y_pred = clf.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    return acc


def decision_tree(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    clf_dec = DecisionTreeClassifier()

    clf_dec.fit(X_train, y_train)

    y_pred = clf_dec.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    return acc


def logistic_regression(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    clf_log = LogisticRegression()

    clf_log.fit(X_train, y_train)

    y_pred = clf_log.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    # saving model
    with open("logistic_regression.pickle", "wb") as f:
         pickle.dump(clf_log, f)

    plt.bar(['1', '2', '3', '4', '5', '6', '7'], f1_sc)
    plt.title('f1 score for linear regression')
    # Saving the plot as a 'png'
    # plt.savefig('f1_score_linreg.png')
    plt.show()

    return acc


def naive_bayes(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    clf_gnb = GaussianNB()

    clf_gnb.fit(X_train, y_train)

    y_pred = clf_gnb.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    return acc


def multilayer_perceptron(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array) -> np.float:

    clf_MLP = MLPClassifier(hidden_layer_sizes=(150, 7), activation='logistic', random_state=1, max_iter=220)

    clf_MLP.fit(X_train, y_train)

    y_pred = clf_MLP.predict(X_test)

    print('y_pred: ', y_pred)
    print('y_test: ', y_test)

    acc = accuracy_score(y_test, y_pred)
    print('acc:', acc)

    f1_sc = f1_score(y_test, y_pred, average=None)
    print('f1 score: ', f1_sc)

    return acc
