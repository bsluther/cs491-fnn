from layersWadam import Layer
import numpy as np
from numpy.typing import NDArray
from functions import LossKey, get_loss_fn

# pylint: disable =unused-argument

# TODO: using default 64-bit floating point numbers, try switching to 16 or 32 to save time.


class History:
    """
    Stores the intermediate values computed during forward and backpropagation which will be needed
    to find the loss-to-weight gradients.
    """

    def __init__(self, x: float | NDArray, K: int):  # pylint: disable=invalid-name
        # Input value
        self.x = x
        # Forward preactivation values
        self.a = [np.array([])] * K
        # Forward postactivation values
        self.h = [np.array([])] * K
        # Loss-to_node gradients for preactivation values
        self.g_a = [np.array([])] * K
        # Loss-to-node gradients for postactivation values
        self.g_h = [np.array([])] * K
        # Computed loss
        self.loss = None


class FNN:
    def __init__(
        self, layers: tuple[Layer, ...], lr=0.01, bias=True, rng=np.random.default_rng(), decay1=.9, decay2=.999,eps=1e-8
    ):
        # Check that layer sizes match up
        if not self.validate_layer_sizes(layers):
            raise ValueError("Layer size mismatch")

        self.lr = lr
        self.rng = rng
        self.layers = layers
        self.bias = bias
        self.dcay1=decay1
        self.dcay2 = decay2
        self.eps = eps
        self.t = 1
        if bias:
            # Add a bias node to each layer by adding a row and column to each each weight matrix
            # (except the last, which will only get an additinal column).
            #
            # That is, we had a m x n matrix, which is a linear map T: R^n -> R^m
            # We want an m+1 x n+1 matrix, which is a linear map T: R^n+1 -> R^m+1

            for i, layer in enumerate(self.layers):
                # Add a column to all the weight matrices, from 0 to 1 because that is standard
                # practice for bias.
                layer.weights = np.append(
                    layer.weights,
                    values=self.rng.uniform(
                        low=0, high=1, size=(layer.weights.shape[0], 1)
                    ),
                    axis=1,
                )

                # Add a row to all the weight matrices except the last, because the output of the
                # last layer is the output of the network and shouldn't include a bias node.
                #
                # Technically, these values are unused because they will only affect the value of
                # the last entry of the output vector, which is the bias node which will be reset to
                # 1 anyway. But, if we leave them out to save time, all our matrix multiplications
                # will be mismatched.
                # So initialize them all to zero, which could be slightly faster than random values
                # if it speeds up the floating point arithmetic.
                if i != len(layers) - 1:
                    layer.weights = layer.weights = np.append(
                        layer.weights,
                        values=np.zeros((1, layer.weights.shape[1])),
                        axis=0,
                    )

    def forward(self, x: float | NDArray):
        """
        Just a convenience wrapper around forward_with_history at this point.

        Args:
            x (float | NDArray): Input value(s).

        Returns:
            NDArray: the output of the network.
        """
        o, _ = self.forward_with_history(x)
        return o

    def forward_with_history(
        self, x: float | NDArray
    ) -> tuple[float | NDArray, History]:
        """
        Compute the output of the network and store intermediate values for use in
        backpropagation.

        Args:
            x (float | NDArray): the input to the network.

        Returns:
            tuple[float | NDArray, History]:
            1. The output the network.
            2. The forward history.
        """
        # Append bias node to the input with value 1 (if bias flag is true)
        _x = np.append(x, 1) if self.bias else x

        # Initialize history to store intermediate values for use in backpropagation
        history = History(_x, len(self.layers))
        # Compute the first layer outside the loop to get things setup
        # Linear combination of weights and inputs
        history.a[0] = self.layers[0].weights @ _x
        # Apply the activation function element-wise, the Layer instance stores the function
        history.h[0] = self.layers[0].activation_forward(history.a[0])

        # Continue forward propagation
        for k in range(1, len(self.layers), 1):
            # Current layer.
            layer = self.layers[k]
            # Linear combination of this layer's weights with the postactivation value of the
            # previous layer.
            history.a[k] = layer.weights @ history.h[k - 1]
            # Element-wise application of activation function to the preactivation values of this
            # layer.
            history.h[k] = layer.activation_forward(history.a[k])

            # Reset the bias node (last node) to 1 in the postactivation values
            if self.bias and k != len(self.layers) - 1:
                history.h[k][-1] = 1

        # The last postactivation value history.h[-1] is the output of the network
        return history.h[-1], history

    def backward_from_history(
        self, y: float | NDArray, history: History, loss_key: LossKey
    ):
        """
        Compute gradients with respect to the loss function specified by the <code>loss_key</code>.
        History is the data structure used to store intermediate values.

        Args:
            y (float | NDArray): the observed value to compute the loss against.
            history (History): a history containing the already computed forward propagation values,
            loss_key (LossKey): the loss function to compute the gradients against, options are
            - "mse"
            - "log"

        Returns:
            _type_: _description_
        """
        # Lookup the derivative of the loss function with the provided key
        # We don't actually need the loss itself for backpropagation so it's not computed
        _, loss_backward = get_loss_fn(loss_key)
        K = len(self.layers)  # pylint: disable=invalid-name

        # Derivative of loss with respect to the output, the far-right term of the gradients
        d_loss = loss_backward(history.h[-1], y)

        # The preactivation values of the last layer
        a_last = history.a[-1]
        # The postactivation values of the last layer
        h_last = history.h[-1]
        # The loss-to-node gradient for the preactivation values of the last layer
        g_a_last = self.layers[-1].activation_backward(a_last, h_last, d_loss)
        # Store the above in the history
        history.g_h[-1] = d_loss
        history.g_a[-1] = g_a_last

        # Initialoize a list to hold the loss-to-weight gradients
        gradients = [np.array([])] * K
        # Compute the loss-to-weight gradient for the last layer
        # warning: this won't work if there are no hidden layers (index will be out of bounds)
        gradients[-1] = np.outer(g_a_last, history.h[-2])

        # The last layer's gradients were already computed, continue backward through the rest
        for k in range(K - 2, -1, -1):
            # Compute the loss-to-node gradient for the postactivation vector of the kth layer
            # by crossing the linear combination W[k+1] @ h[k]
            # I used a dot product here because it's possible we get a scalar value in the output
            # layer's gradient (if there is only 1 output node). Numpy will treat the dot product of
            # two matrices as a matrix multiplication.
            history.g_h[k] = np.dot(self.layers[k + 1].weights.T, history.g_a[k + 1])

            # Compute the loss-to-node gradient for the preactivation vector of the kth layer by
            # crossing the element-wise application of the activation function, phi(a[k]).
            history.g_a[k] = self.layers[k].activation_backward(
                history.a[k], history.h[k], history.g_h[k]
            )

            # Compute the loss-to-weight gradient
            # When we reach the first layer, need to pass the input x rather than look up h[k-1].
            prev_h = history.h[k - 1] if k > 0 else np.array([history.x])
            gradients[k] = np.outer(history.g_a[k], prev_h)

        # Return the loss-to-weight derivatives and history (probably don't need to return history).
        return gradients, history

    def gd(self, x: float | NDArray, y: float | NDArray, loss_key: LossKey):
        """
        Perform gradient descent by adjusting the weights according the their gradients with
        respect to the loss.

        Args:
            x (float | NDArray): the input to the network.
            y (float | NDArray): the observed value to compute the loss against.
            loss_key (LossKey): loss function to use in determining the gradients, options are:
            - "mse"
            - "log"
        """
        # Compute the forward values and store intermediate values
        y_hat, history = self.forward_with_history(x)

        if loss_key == "mse":
            # Compute the mse loss value
            loss = 0.5 * np.mean((y_hat - y) ** 2)
        elif loss_key == "log":
            loss = loss = np.log(1 + np.exp(-y * y_hat))
        elif loss_key == "nll":
            loss = -y_hat[y]  # Index into the log-probability for the correct class



        # Compute the loss-to-weight gradients via backpropagation
        gradients, history = self.backward_from_history(y, history, loss_key)
        # Update the weights according to the gradients
        for k, gradient in enumerate(gradients):
            self.layers[k].ADAM_up(gradient,self.lr,self.dcay1,self.dcay2,self.eps,self.t)
        self.t = 1+self.t

        return loss

    def minibatchGD(self, X_batch, y_batch, loss_key):
        """
        Perform forward and backward pass on a mini-batch and update weights.

        Args:
            X_batch (NDArray): Input data for the mini-batch (shape: (batch_size, input_dim)).
            y_batch (NDArray): True labels for the mini-batch (shape: (batch_size,)).
            loss_key (str, optional): Loss function key (e.g., "nll", "mse")

        Returns:
            float: Average loss over the mini-batch.
        """
        batch_size = X_batch.shape[0]
        accumulated_gradients = [np.zeros_like(layer.weights) for layer in self.layers]
        total_loss = 0.0

        for i in range(batch_size):
            x = X_batch[i]
            y = y_batch[i]

            # Forward pass
            y_hat, history = self.forward_with_history(x)  # Shape: (batch_size, num_classes)

            # Compute loss
            if(loss_key == "nll"):
                loss = -y_hat[y]  # Scalar
            # else:

            # ( from the slides add up all the losses )
            total_loss += loss

            # Backward pass to compute gradients
            gradients, history = self.backward_from_history(y, history, loss_key)  # List of gradients

            for j in range(len(accumulated_gradients)):
                accumulated_gradients[j] += gradients[j]

        # Average the gradients and loss over the batch
        averaged_gradients = [g_acc / batch_size for g_acc in accumulated_gradients]
        average_loss = total_loss / batch_size

        # Update weights
        for k, gradient in enumerate(averaged_gradients):
            self.layers[k].ADAM_up( gradient, self.lr, self.dcay1, self.dcay2, self.eps, self.t)
        self.t = 1+self.t
        return average_loss

    @staticmethod
    def validate_layer_sizes(layers: tuple[Layer, ...]):
        """
        Check that layer sizes match up: output size of layer k must be the same as input size of
        layer k + 1.

        Args:
            layers (tuple[Layer, ...]): The ordered Layers which make up the network.

        Returns:
            bool: True if the layers match up, otherwise False.
        """
        # Must have at least one layer
        if len(layers) == 0:
            return False

        prev_out_features = layers[0].weights.shape[0]

        # Output size of layer k must be the same as input size of layer k + 1
        for layer in layers[1:]:
            curr_in_features = layer.weights.shape[1]
            if curr_in_features != prev_out_features:
                return False
            prev_out_features = layer.weights.shape[0]

        return True
