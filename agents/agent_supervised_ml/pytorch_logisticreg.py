import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from agents.agent_supervised_ml.data_prep import clean_scores


def logistic_regression():

    # prepare data
    """x_numpy = np.array([[4, 6, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 6, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 5, 4, 1, 7, 7, 3, 2, 2, 5, 5, 7, 4, 3, 6, 6, 1, 0, 0, 0, 0, 0, 0, 0],
                        [7, 1, 1, 2, 3, 6, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [2, 1, 5, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 3, 5, 1, 5, 6, 7, 4, 2, 1, 5, 5, 6, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 7, 7, 3, 5, 2, 6, 5, 4, 5, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [6, 3, 1, 7, 5, 4, 5, 5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [4, 3, 6, 7, 4, 2, 2, 2, 7, 4, 6, 4, 5, 3, 1, 7, 6, 6, 1, 7, 1, 1, 2, 1],
                        [7, 6, 3, 4, 5, 4, 1, 2, 7, 2, 5, 3, 2, 2, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0]])

    y_numpy = np.array([[-2], [0], [2], [-2], [0], [2], [6], [4], [9], [-10]])
    """
    X, y = clean_scores()

    n_samples, n_features = X.shape

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # scale
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train = torch.from_numpy(X_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    # 1) Model
    # Linear model f = wx + b , sigmoid at the end
    class Model(nn.Module):
        def __init__(self, n_input_features):
            super(Model, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            y_pred = torch.sigmoid(self.linear(x))
            return y_pred

    model = Model(n_features)

    # 2) Loss and optimizer
    num_epochs = 200
    learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # 3) Training loop
    for epoch in range(num_epochs):
        # Forward pass and loss
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # zero grad before new step
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
        print(f'accuracy: {acc.item():.4f}')

    """
    # then scale X data?? how?

    X = torch.from_numpy(x_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))

    # y = y.view(y.shape[0], 1)

    n_samples, n_features = X.shape

    # model
    class LogisticRegression(nn.Module):
        def __init__(self, n_input_features):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(n_input_features, 1)

        def forward(self, x):
            y_predicted = torch.sigmoid(self.linear(x))
            return y_predicted

    model = LogisticRegression(n_features)

    # loss and optimizer
    learning_rate = 0.01
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # forward pass and loss
        y_predicted = model(X)
        loss = criterion(y_predicted, y)

        # backward pass
        loss.backward()

        # update
        optimizer.step()
        optimizer.zero_grad()

        if (epoch+1) % 10 == 0:
            print(f'epoch: {epoch}, loss = {loss.item()}:4f')

    # with torch.no_grad():
    #    y_predicted = model(X)
    #   y_predicted_cls = y_predicted.round()
    #   acc = y_predicted_cls.eq(y_test).sum() / float(y.test.shape[0])
    #   print( acc...

    # plot
    predicted = model(X).detach().numpy()
    plt.plot(x_numpy, y_numpy, 'ro')
    plt.plot(x_numpy, predicted, 'b')
    plt.show()
    """
