import time
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import random
from fnn2 import FNN
from layers2 import Layer

# Load MNIST dataset using sklearn
def load_iris1(test_size=0.2, random_state=42):
    iris = load_iris()
    # mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = iris['data'].astype(np.float32)
    y = iris['target'].astype(int)  # Convert targets to integers
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

# Training function for FNN with time cost, accuracy, and loss convergence tracking
def train_fnn_Iris(X_train, y_train, X_test, y_test, batch_size=128, epochs=20, learning_rate=0.01):
    rng = np.random.default_rng(1337)

    # Adam hyperparameters
    beta1, beta2 = 0.9, 0.999
    epsilon = 1e-8
    t = 1  # time step for Adam algorithm


    #initialize the FNN
    l1 = Layer(in_features=4, out_features=16, activation_key="relu", use_xavier=True)
    l2 = Layer(in_features=16, out_features=16, activation_key="relu", use_xavier=True)
    l3 = Layer(in_features=16, out_features=3, activation_key="log_softmax", use_xavier=True)
   # l4 = Layer(in_features=64, out_features=10, activation_key="log_softmax", use_xavier=True)

    fnn = FNN(layers=(l1, l2, l3), lr=learning_rate, bias=False, rng=rng)
    history = []  # Store loss for each epoch
    train_accuracies = []

    start_time = time.time()  # Start timer
    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)  # Shuffle data every epoch
        epoch_loss = 0.0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            loss, gradients = fnn.minibatchGD(X_batch, y_batch, loss_key='nll')
            epoch_loss += loss

            # Apply Adam update for each layer
            for j, layer in enumerate(fnn.layers):
                layer.ADAM_up(gr=gradients[j], lr=learning_rate, b1=beta1, b2=beta2, eps=epsilon, t=t)

            t += 1  # Increment time step for Adam

        avg_epoch_loss = epoch_loss / (len(X_train) / batch_size)
        history.append(avg_epoch_loss)

        # Calculate accuracy on test data
        accuracy = evaluate_network(fnn, X_test, y_test, batch_size)
        train_accuracies.append(accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    total_time = time.time() - start_time  # Calculate time cost
    print(f"Training completed in {total_time:.2f} seconds")
    return fnn, history, total_time, train_accuracies

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

def plot_loss_accuracy(history, accuracies):
    epochs = range(1, len(history) + 1)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs, accuracies, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Loss and Accuracy Over Epochs')
    fig.tight_layout()
    plt.show()

def main():
    # Load and preprocess MNIST dataset
    X_train, X_test, y_train, y_test = load_iris1()

    # Train the FNN on MNIST dataset
    print("Starting training on Iris Dataset...")
    fnn, history, total_time, train_accuracies = train_fnn_Iris(X_train, y_train, X_test, y_test, batch_size=16, epochs=250, learning_rate=0.001)
    print("Training complete.")

    # Print the total training time and final accuracy
    final_accuracy = evaluate_network(fnn, X_test, y_test, batch_size=16)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Plot the loss convergence
    plot_loss_convergence(history)
    plot_loss_accuracy(history, train_accuracies)

if __name__ == "__main__":
    main()
