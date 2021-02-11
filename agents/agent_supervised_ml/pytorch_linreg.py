import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from agents.agent_supervised_ml.data_prep import clean_scores


def linear_regression():

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
    x_numpy, y_numpy = clean_scores()
    print(x_numpy.shape)
    # print(x_numpy)
    # print(y_numpy)

    X = torch.from_numpy(x_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))

    # y = y.view(y.shape[0], 1)

    n_samples, n_features = X.shape

    # model
    input_size = n_features
    output_size = 1
    model = nn.Linear(input_size, output_size)

    # loss and optimizer
    learning_rate = 0.01
    criterion = nn.MSELoss()
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
            print(epoch, loss.item())

    # plot
    predicted = model(X).detach().numpy()
    plt.plot(x_numpy, y_numpy, 'ro')
    plt.plot(x_numpy, predicted, 'b')
    plt.show()
