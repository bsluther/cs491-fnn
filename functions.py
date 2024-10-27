from typing import Literal
from collections import namedtuple
from numpy.typing import NDArray
import numpy as np

# Activation and loss functions paired with their derivatives, retrievable by key.

###### Activation Functions

# pylint: disable=unused-argument
# Different backward functions need different data, so in order to presen a common interface, some
# functions have unused parameters.

# String literal type containing the implemented activation functions
ActivationKey = Literal["relu", "sigmoid", "identity", "log_softmax"]

# Tuple type to store functions and their derivatives consistently
Fn = namedtuple("Fn", ["forward", "backward"])

# Parameters for activation functions:
# a_curr: the preactivation vector for the current layer
# h_curr: the postactivation vector for the current layer
# g_next: the next gradient, which in the case of an activation function is the loss-to-node
# gradient for the pre-activation vector, g_a_next.

#       Linear combination       Activation         Linear combination
#         W_curr @ h_prev        phi(a_curr)         W_next @ h_curr
#   h_prev     ---->     a_curr     ---->     h_curr     ---->     a_next
#  g_h_prev    <----    g_a_curr    <----    g_h_curr    <----    g_a_next
#


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


def log_softmax_forward(a_curr: np.ndarray) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    a_shifted = a_curr - np.max(a_curr, axis=-1, keepdims=True)
    # Compute log-softmax
    log_probs = a_shifted - np.log(np.sum(np.exp(a_shifted), axis=-1, keepdims=True))
    return log_probs


def log_softmax_backward(
    a_curr: np.ndarray,
    h_curr: np.ndarray,
    g_next: np.ndarray,
) -> np.ndarray:
    # Compute softmax probabilities
    softmax_probs = np.exp(h_curr)
    # Compute gradient
    grad_a_curr = g_next - np.dot(g_next, softmax_probs) * softmax_probs
    return grad_a_curr


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
    if key == "log_softmax":
        return Fn(log_softmax_forward, log_softmax_backward)

    raise ValueError(f"Unrecognized activation key '{key}'")


##### Loss Functions

# y_hat: output of the network
# y: observed value to use in computing the loss

# String literal type containing the implemented loss functions
LossKey = Literal["nll", "mse", "log_example"]


# def mse_forward(o: float | NDArray, y: float | NDArray):
#     return (y - o) ** 2
def mse_forward(y_hat: float | NDArray, y: float | NDArray):
    return 0.5 * np.mean((y_hat - y) ** 2)


def mse_backward(y_hat: float | NDArray, y: float | NDArray):
    return 2 * (y_hat - y)


def nll_loss_forward(y_hat: np.ndarray, y: np.ndarray) -> float:
    # Extract the log probability of the correct class
    log_prob = y_hat[y]
    # Compute the negative log likelihood loss
    loss = -log_prob
    return loss


def nll_loss_backward(y_hat: NDArray, y: NDArray):
    softmax_probs = np.exp(y_hat)  # Convert log probabilities to probabilities
    y_one_hot = np.zeros_like(y_hat)
    y_one_hot[y] = 1
    delta = softmax_probs - y_one_hot
    return delta


# Note: this log_loss was just used to mimmick the book example, it doesn't actually incorporate the
# observed value y so probably not actually useful for our purposes.
def log_loss_example_forward(y_hat: float | NDArray, y: float | NDArray):
    return -np.log(y_hat)


def log_loss_example_backward(y_hat: float | NDArray, y: float | NDArray):
    return -1 / y_hat


def get_loss_fn(key: LossKey):
    if key == "log":
        return Fn(log_loss_example_forward, log_loss_example_backward)
    if key == "mse":
        return Fn(mse_forward, mse_backward)
    if key == "nll":
        return Fn(nll_loss_forward, nll_loss_backward)
    raise ValueError(f"Unrecognized loss key '{key}'")


# Stashing this in case it's needed
# def log_loss_from_gd(o: NDArray, y: NDArray):
#     return np.log(1 + np.exp(-y * o))
