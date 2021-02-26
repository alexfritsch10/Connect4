from typing import Tuple

import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

class Model:

    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(126 * 5), activation='logistic', random_state=1, max_iter=1500,
                                   alpha=0.001, n_iter_no_change=10, learning_rate='adaptive', learning_rate_init=0.005,
                                   shuffle=True, tol=0.0001, verbose=0)

    def train(self, dataset):
        X = np.array([])
        y = np.array([])
        for data in dataset:
            X = np.append(data[1], X)
            y = np.append(data[0], y)

        X = X.reshape((-1, 42))

        X, y = self.clean_data(X, y)

        X = OneHotEncoder(categories=[[0, 1, 2]] * 42).fit_transform(X).toarray().astype(np.int8)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
        print("Input: ")
        print(X_train[:5])
        print("Output: ")
        print(y_train[:5])

        #X = np.array(inputData).reshape((-1, self.numberOfInputs))
        #enc = OneHotEncoder(handle_unknown='ignore')

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        print('y_pred: ', y_pred)
        print('y_test: ', y_test)
        # y_test = y_test.flatten()
        # print(y_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        print('acc:', acc)

        f1_sc = f1_score(y_test, y_pred, average=None)
        print('f1 score: ', f1_sc)



    def predict(self, data):
        print("Want to predict data: ")
        print(data)
        print("Reshaped data: ")
        reshapedData = np.array(data).reshape(-1, 42)
        print(reshapedData)
        predData = OneHotEncoder(categories=[[0, 1, 2]] * 42).fit_transform(reshapedData).toarray().astype(np.int8)
        print("Getting right data")
        print(self.model.predict(predData)[0])

        return self.model.predict(predData)[0]

    def clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes duplicates from X and assigns most frequent label from y.
        Parameters
        ----------
        X : np.ndarray
            Data in shape(n_samples, n_features)
        y : np.ndarray
            Labels in shape(n_samples, )
        Returns
        -------
        Cleaned X and y
        """
        # unique boards and inverse index mapping
        X_new, idx_inv = np.unique(X, axis=0, return_inverse=True)
        # init new y
        y_new = np.zeros((X_new.shape[0],), dtype=np.int8)
        for i in range(y_new.shape[0]):
            # moves corresponding to unique boards
            y_to_unique_entry = y[idx_inv == i]
            # frequency of moves
            columns, counts = np.unique(y_to_unique_entry, return_counts=True)
            # most frequent move
            y_new[i] = columns[np.argmax(counts)].astype(np.int8)
        return X_new, y_new