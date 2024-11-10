import time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from fnn2 import FNN
from layers2 import Layer

# Load MNIST dataset using sklearn
def load_mnist(test_size=0.2, random_state=42):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].astype(np.float32) / 255.0  # Normalize the input to [0, 1]
    y = mnist['target'].astype(int)  # Convert targets to integers
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Training function for FNN with time cost, accuracy, and loss convergence tracking
def train_fnn_mnist(X_train, y_train, X_test, y_test, batch_size=128, epochs=20, learning_rate=0.01):
    rng = np.random.default_rng(1337)
    # Initialize the FNN with LogSoftmax activation in the last layer
    l1 = Layer(in_features=784, out_features=128, activation_key="relu", use_xavier=True)
    l2 = Layer(in_features=128, out_features=64, activation_key="relu", use_xavier=True)
    l3 = Layer(in_features=64, out_features=64, activation_key="relu", use_xavier=True)
    l4 = Layer(in_features=64, out_features=10, activation_key="log_softmax", use_xavier=True)

    fnn = FNN(layers=(l1, l2, l3, l4), lr=learning_rate, bias=True, rng=rng)
    history = []  # Store loss for each epoch

    start_time = time.time()  # Start timer
    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)  # Shuffle data every epoch
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss = fnn.minibatchGD(X_batch, y_batch, loss_key='nll')
            epoch_loss += loss
        avg_epoch_loss = epoch_loss / (len(X_train) / batch_size)
        history.append(avg_epoch_loss)

        # Calculate accuracy on test data
        accuracy = evaluate_network(fnn, X_test, y_test, batch_size)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    total_time = time.time() - start_time  # Calculate time cost
    print(f"Training completed in {total_time:.2f} seconds")
    return fnn, history, total_time

def evaluate_network(fnn, X_test, y_test, batch_size=128):
    correct = 0
    total = 0
    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]
        for x, y_true in zip(X_batch, y_batch):
            output = fnn.forward(x)
            predicted = np.argmax(output)
            correct += int(predicted == y_true)
            total += 1
    accuracy = correct / total
    return accuracy

def plot_loss_convergence(history):
    plt.plot(history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Convergence Over Epochs")
    plt.legend()
    plt.show()

def main():
    # Load and preprocess MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist()

    # Train the FNN on MNIST dataset
    print("Starting training...")
    fnn, history, total_time = train_fnn_mnist(X_train, y_train, X_test, y_test, batch_size=64, epochs=9, learning_rate=0.001)
    print("Training complete.")

    # Print the total training time and final accuracy
    final_accuracy = evaluate_network(fnn, X_test, y_test, batch_size=64)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Plot the loss convergence
    plot_loss_convergence(history)

if __name__ == "__main__":
    main()
