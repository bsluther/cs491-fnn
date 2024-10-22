import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from fnn import FNN


def test_fnn_sin():
    rng = np.random.default_rng(1337)
    l1 = Layer(1, 16, "relu", rng=rng)
    l2 = Layer(16, 16, "sigmoid", rng=rng)
    l3 = Layer(16, 1, "identity", rng=rng)
    net = FNN((l1, l2, l3), lr=0.01, bias=True, rng=rng)
    x_train = np.linspace(-3, 3, 100)
    y_train = np.sin(x_train)
    loss_key = "mse"
    history = []
    epochs = 1000
    for _ in range(epochs):
        epoch_loss = 0
        for x, y in zip(x_train, y_train):
            loss = net.gd(x, y, loss_key)
            epoch_loss += loss
        average_loss =  epoch_loss / len(x_train)
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
    plt.xlabel('Epochs')
    if loss_key == "mse":
        plt.ylabel('Mean Squared Error')
    elif loss_key == "log":
        plt.ylabel('Log Error')
    plt.show()



test_fnn_sin()


# Debugging example that mirrors the example in the book (page 54)
def test_book_example():
    x = np.array([2, 1, 2])
    W_1 = np.array([[2, -2, 0, 0], [-1, 5, -1, 0], [0, 3, -2, 0], [0, 0, 0, 0]])
    W_2 = np.array([-1, 1, -3, 0])

    l1 = Layer(3, 3, "relu")
    l2 = Layer(3, 1, "sigmoid")

    net = FNN((l1, l2))
    net.layers[0].weights = W_1
    net.layers[1].weights = W_2

    o, hist = net.forward_with_history(x)
    M, history = net.backward_from_history(1, hist, "log")


# test_book_example()
