from typing import Literal
from collections import namedtuple
from numpy.typing import NDArray
import numpy as np

# Activation and loss functions paired with their derivatives, retrievable by key.

# Activation Functions

# pylint: disable=unused-argument
# Different backward functions need different data, so in order to present
# a common interface, some functions have unused parameters.


ActivationKey = Literal["relu", "sigmoid", "identity"]

Fn = namedtuple("Fn", ["forward", "backward"])


def identity_forward(a_curr: NDArray):
    return a_curr


def identity_backward(
    a_curr: NDArray,
    h_curr: NDArray,
    g_next: NDArray,
):
    return g_next


def sigmoid_forward(a_curr: NDArray):
    return 1 / (1 + np.exp(-a_curr))


def sigmoid_backward(
    a_curr: NDArray,
    h_curr: NDArray,
    g_next: NDArray,
):
    return g_next * h_curr * (1 - h_curr)


def relu_forward(a_curr):
    return np.where(a_curr > 0, a_curr, 0)


def relu_backward(
    a_curr: NDArray,
    a_next: NDArray,
    g_next: NDArray,
):
    return np.where(a_curr > 0, g_next, 0)


def get_activation_fn(
    key: ActivationKey,
):
    """
    Get an activation function (forward) and it's derivative (backward) by key.

    Args:
        key (str): the activation function to lookup, options are:
        - "relu"
        - "sigmoid"
        - "identity

    Raises:
        ValueError: if the provided key is not one of the implemented activations.

    Returns:
        ActivationFn: the forward function and its derivative (backward).
    """
    if key == "relu":
        return Fn(relu_forward, relu_backward)
    if key == "sigmoid":
        return Fn(sigmoid_forward, sigmoid_backward)
    if key == "identity":
        return Fn(identity_forward, identity_backward)
    raise ValueError(f"Unrecognized activation key '{key}'")


# Loss Functions

LossKey = Literal["log", "mse"]


def log_loss_forward(o: float | NDArray, y: float | NDArray):
    return -np.log(o)


def log_loss_backward(o: float | NDArray, y: float | NDArray):
    return -1 / o


def mse_forward(o: float | NDArray, y: float | NDArray):
    return (y - o) ** 2


def mse_backward(o: float | NDArray, y: float | NDArray):
    return 2 * (o - y)


def get_loss_fn(key: LossKey):
    if key == "log":
        return Fn(log_loss_forward, log_loss_backward)
    if key == "mse":
        return Fn(mse_forward, mse_backward)
    raise ValueError(f"Unrecognized loss key '{key}'")
