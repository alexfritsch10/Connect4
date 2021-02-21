import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from agents.agent_supervised_ml_2.gameController import generate_data_set


class Model:

    def __init__(self, numberOfInputs: int, numberOfOutputs: int, batchSize: int, epochs: int):
        self.model = MLPClassifier(hidden_layer_sizes=(126 * 5), activation='logistic', random_state=1, max_iter=1500,
                                   alpha=0.001, n_iter_no_change=10, learning_rate='adaptive', learning_rate_init=0.005,
                                   shuffle=True, tol=0.0001, verbose=0)

    def train(self, dataset):
        input = []
        output = []
        for data in dataset:
            input.append(data[1])
            output.append(data[0])

        X = np.array(input).reshape((-1, self.numberOfInputs))
        y = to_categorical(output, num_classes=3)
        limit = int(0.8 * len(X))
        X_train = X[:limit]
        X_test = X[limit:]
        y_train = y[:limit]
        y_test = y[limit:]
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=self.epochs,
                       batch_size=self.batchSize)

    def predict(self, data, index: int):
        return self.model.predict(np.array(data).reshape(-1, self.numberOfInputs))[0][index]