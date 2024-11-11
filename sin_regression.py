import numpy as np
import matplotlib.pyplot as plt
from layers2 import Layer
from fnn2 import FNN


def test_fnn_sin():
    rng = np.random.default_rng(1337)
    l1 = Layer(1, 30, "identity", rng=rng)
    l2 = Layer(30, 20, "sigmoid", rng=rng)
    l3 = Layer(20, 1, "identity", rng=rng)
    net = FNN((l1, l2, l3), lr=0.01, bias=True, rng=rng)
    x_train = np.linspace(-3, 3, 100)
    y_train = np.sin(x_train)
    loss_key = "mse"
    history = []
    epochs = 100
    for _ in range(epochs):
        epoch_loss = 0
        for x, y in zip(x_train, y_train):
            # loss = net.gd(x, y, loss_key)
            loss = net.newton_bfgs(x, y, loss_key)
            epoch_loss += loss
        average_loss = epoch_loss / len(x_train)
        history.append(average_loss)

    y_hats = np.array([])
    for x in x_train:
        o, _ = net.forward_with_history(x)
        y_hats = np.append(y_hats, o)
    plt.figure(1)
    plt.plot(x_train, y_train, "b.")
    plt.plot(x_train, y_hats, "r.")
    plt.figure(2)
    plt.plot(history)
    plt.xlabel("Epochs")
    if loss_key == "mse":
        plt.ylabel("Mean Squared Error")
    plt.show()


test_fnn_sin()
