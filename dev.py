import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from fnn import FNN


def test_fnn_sin():
    rng = np.random.default_rng(1337)
    l1 = Layer(1, 10, "relu", rng=rng)
    l2 = Layer(10, 10, "relu", rng=rng)
    l3 = Layer(10, 10, "relu", rng=rng)
    l4 = Layer(10, 10, "relu", rng=rng)
    l5 = Layer(10, 1, "identity", rng=rng)
    net = FNN((l1, l2, l3, l4, l5), lr=0.001, bias=True, rng=rng)
    x_train = np.linspace(-4, 4, 100)
    y_train = np.sin(x_train)

    epochs = 1500
    for _ in range(epochs):
        for x, y in zip(x_train, y_train):
            net.gd(x, y, "mse")

    y_hats = np.array([])
    for x in x_train:
        o, _ = net.forward_with_history(x)
        y_hats = np.append(y_hats, o)
    plt.plot(x_train, y_train, "b.")
    plt.plot(x_train, y_hats, "r.")

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
