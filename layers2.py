import numpy as np
from functions import get_activation_fn
from typing import Literal
from functions import ActivationKey

# Modification the primary Layer to use ADAM learning.


class Layer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_key: ActivationKey,
        rng=np.random.default_rng(),
    ):
        """
        Construct a Layer with <code>in_features</code> input features and
        <code>out_features</code> outputs using an activation function which
        will be looked up according the provided key.

        Args:
            in_features (int):
                The number of input features.

            out_features (int):
                The number of outputs this layer produces.

            activation_key(str):
                A string indicating the activation function to use in this layer. Options are:
                - "relu"
                - "sigmoid"

            rng(generator):
                A random number generator which will be used to initialize the weights.
                Defaults to a random number generator with no seed value.

        Raises:
            ValueError: when the provided <code>activation_key</code> does not match a known
            activation function.
        """
        forward, backward = get_activation_fn(activation_key)
        self.in_features = in_features
        self.out_features = out_features
        self.activation_key = activation_key
        self.activation_forward = forward
        self.activation_backward = backward
        self.rng = rng

        if activation_key in ["relu", "leaky_relu", "log_softmax"]:
            self.weights = self.rng.normal(
                0, np.sqrt(2 / in_features), size=(out_features, in_features)
            )
        else:
            self.weights = self.rng.uniform(
                low=-1, high=1, size=(out_features, in_features)
            )

        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        # print(
        #     f" Initial max weight = {np.max(np.abs(self.weights))}"
        # )

    def ADAM_up(self, gr, lr, b1, b2, eps, t):
        """
        gr is the current gradient,
        lr is learning rate,
        b1 is the decay rate is a hyperameter for the average decaying rate each iteration, gives control on how much the past average effects the current
        gradient
        b2 is the decay for the larger gradients int he ast that would have a larger impact.
        eps prevent divide by zero error, and t is the time step for bias correction
        """
        #help with gradient explosion
        clip_value = 1.0
        gr = np.clip(gr, -clip_value, clip_value)
        # updating each parameter first
        self.m = b1 * self.m + (1 - b1) * gr
        self.v = b2 * self.v + (1 - b2) * (gr**2)
        # creating bias correction for each method for update, this because we are initakizing to zero. This will help deal with the bias towards 0
        m_correction = self.m / (1 - (b1**t))
        v_correction = self.v / (1 - (b2**t))
        # now doing the weight update with adam
        self.weights = self.weights - lr * (
            m_correction / ((np.sqrt(v_correction)) + eps)
        )
    #nestrov learning technique update
    def NestML(self, lr, gr, b1):
        pass
    #newton update technique
    def NewtUpdate():
        pass
