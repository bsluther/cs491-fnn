import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from fnn import FNN


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
    M, history = net.backward_from_history(1, hist, "log_example")


# test_book_example()
