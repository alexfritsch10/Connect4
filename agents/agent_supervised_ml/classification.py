import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from agents.agent_supervised_ml.data_prep import clean_scores


def linear_svm():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print('(samples, number of features): ', X_train.shape)

    # clf_svm = svm.SVC(kernel='linear')
    clf_svm = svm.SVC()

    clf_svm.fit(X_train, y_train)

    y_pred = clf_svm.predict(X_test)

    print('y_pred: ')
    print(y_pred)
    # print('y_test: ')
    # print(y_test)
    y_test = y_test.flatten()
    print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:')
    print(acc)


def k_nearest_neighbours():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print('(samples, number of features): ', X_train.shape)

    # clf_svm = svm.SVC(kernel='linear')
    clf = KNeighborsClassifier(n_neighbors=15)

    clf.fit(X_train, y_train)

    print(X_test)
    print(X_test[1])

    y_pred_single = clf.predict([X_test[1]])
    print(y_pred_single)
    y_pred = clf.predict(X_test)

    print('y_pred: ')
    print(y_pred)
    # print('y_test: ')
    # print(y_test)
    y_test = y_test.flatten()
    print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:')
    print(acc)


def decision_tree():

    X, y = clean_scores()
    # n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    print('(samples, number of features): ', X_train.shape)

    # clf_svm = svm.SVC(kernel='linear')
    clf_dec = DecisionTreeClassifier()

    clf_dec.fit(X_train, y_train)

    y_pred = clf_dec.predict(X_test)

    print('y_pred: ')
    print(y_pred)
    # print('y_test: ')
    # print(y_test)
    y_test = y_test.flatten()
    print(y_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print('acc:')
    print(acc)