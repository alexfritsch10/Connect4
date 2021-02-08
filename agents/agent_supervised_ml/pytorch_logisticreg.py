import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def logistic_regression():

    # prepare data
    x_numpy = np.array([[4, 6, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

    # actually should be split in X_train, x_test, y_train, y_test (test size 20%)

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