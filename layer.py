import numpy as np
from functions import get_activation_fn
from typing import Literal
from functions import ActivationKey


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
        self.weights = self.rng.uniform(
            low=-1, high=1, size=(out_features, in_features)
        )
