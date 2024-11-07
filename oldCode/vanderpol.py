import numpy as np
import matplotlib.pyplot as plt
from layer import Layer
from fnn import FNN

# Creating a vanderpol function whose ODE is given as:
def vanderpol(x1, x2):
    dx1 = x2
    dx2 = -x1 + (1 - x2**2) * x2
    return dx1, dx2

# Defining a function which generates the dataset of states and their next steps which is calculated using a given time step.
def generate_vanderpol_samples(num_samples=1000, time_step=0.5):
    # It generates random number
    rng = np.random.default_rng(1337)
    # It randomly generates initial states
    x_train = rng.uniform(-3, 3, size=(num_samples, 2))
    # It will hold for next step
    y_train = np.zeros_like(x_train)

    for i in range(num_samples):
        x1, x2 = x_train[i]
        dx1, dx2 = vanderpol(x1, x2)

        # Using Euler's method for calculating next state
        x1_next = x1 + time_step * dx1
        x2_next = x2 + time_step * dx2
        y_train[i] = [x1_next, x2_next]

    # print(f"x_train:{x_train.shape}, y_train: {y_train.shape}")
    return x_train, y_train

# Defining a function which trains on vanderpol using feedforward neural network
def test_fnn_vanderpol():
    x_train, y_train = generate_vanderpol_samples(num_samples=1000)

    # Using three different activation function to initialize FNN with three layers
    rng = np.random.default_rng(1337)
    l1 = Layer(2, 20, "relu", rng=rng)
    l2 = Layer(20, 16, "sigmoid", rng=rng)
    l3 = Layer(16, 2, "identity", rng=rng)

    net = FNN((l1, l2, l3), lr=0.01, bias=True, rng=rng)

    # Defining the parameters for training for mini-batch
    batch_size = 32
    num_epochs = 1000
    loss_key = "mse"
    history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        indices = np.random.permutation(len(x_train))
        x_train_shuffled, y_train_shuffled = x_train[indices], y_train[indices]

        # Training Mini-Batch
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            # Using minibatch gradient descent in order to update the weights and compute the average loss.
            batch_loss = net.minibatchGD(batch_x, batch_y, loss_key="mse")
            epoch_loss += batch_loss

        history.append(epoch_loss / (len(x_train) // batch_size))

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss:{history[-1]:.6f}')

    # Plotting the loss over epochs
    plt.figure()
    plt.plot(history)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Vanderpol Training Loss')
    plt.show()

    # Generating test data for evaluating the model
    x_test, y_test_true = generate_vanderpol_samples(num_samples=100)

    # Predicting the next state for test data using the trained FNN
    y_test_pred = np.array([net.forward(x) for x in x_test])

    # Plotting the states for predicted versus true
    plt.figure()
    plt.scatter(y_test_true[:, 0], y_test_true[:, 1], color='blue', label='True')
    plt.scatter(y_test_pred[:, 0], y_test_pred[: , 1], color='red', label='Predicted')

    # Draw connecting lines between corresponding points
    for i in range(len(y_test_true)):
        plt.plot([y_test_true[i, 0], y_test_pred[i, 0]],[y_test_true[i, 1], y_test_pred[i, 1]],
             color='gray', linestyle='-', linewidth=0.5)
    plt.xlabel('x1_next')
    plt.ylabel('x2_next')
    plt.legend()
    plt.title("Predicted Versus True Next State")
    plt.show()

    # It will print the result for first five test
    for i in range(5):
        print(f"Test {i+1}:")
        print(f" True next state: {y_test_true[i]}")
        print(f" Predicted next state: {y_test_pred[i]}")

# Executing the main function
if __name__ == "__main__":
    test_fnn_vanderpol()
