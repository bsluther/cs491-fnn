import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from fnn import FNN  # Assuming you have your FNN class
from layer import Layer  # Assuming your Layer class is implemented
from functions import get_loss_fn  # Implemented NLLLoss and activations

# Load MNIST dataset using sklearn
def load_mnist(test_size = 0.001, random_state = 42):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False) # get the data set
    X = mnist['data'].astype(np.float32) / 255.0  # Normalize the input to [0, 1]
    y = mnist['target'].astype(int)  # Convert targets to integers

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


# Training function for FNN using mini-batch SGD
def train_fnn_mnist(X_train, y_train, X_test, y_test, batch_size=64, epochs=10, learning_rate=0.0001):

    rng = np.random.default_rng(1337)
    # Initialize the FNN with LogSoftmax activation in the last layer
    l1 = Layer(in_features=784, out_features=128, activation_key="relu")
    l2 = Layer(in_features=128, out_features=64, activation_key="sigmoid")
    l3 = Layer(in_features=64, out_features=10, activation_key="log_softmax")

    # Create the FNN model
    fnn = FNN(layers=(l1, l2, l3), lr=learning_rate, bias=True, rng=rng)

    history = []

    for epoch in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)  # Shuffle data every epoch
        epoch_loss = 0.0

        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            batch_loss = 0
            for x, y in zip(X_batch, y_batch):

                # Forward and backward pass for each mini-batch
                loss = fnn.gd(x, y, loss_key="nll")  # Using NLLLoss (log loss)
                batch_loss += loss

            epoch_loss += batch_loss / len(X_batch)

        history.append(epoch_loss / (len(X_train) // batch_size))
        # Evaluate the network after each epoch
        accuracy = evaluate_network(fnn, X_test, y_test, batch_size)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Test Accuracy: {accuracy:.4f}')

def evaluate_network(fnn, X_test, y_test, batch_size=64):
    correct = 0
    total = 0

    for i in range(0, len(X_test), batch_size):
        X_batch = X_test[i:i + batch_size]
        y_batch = y_test[i:i + batch_size]

        # Process each sample individually
        for x, y_true in zip(X_batch, y_batch):

            output = fnn.forward(x)
            predicted = np.argmax(output)
            correct += int(predicted == y_true)
            total += 1

    accuracy = correct / total
    return accuracy

def main():
    # Load and preprocess MNIST dataset
    X_train, X_test, y_train, y_test = load_mnist()

    # Train the FNN on MNIST dataset
    print("Starting training...")
    train_fnn_mnist(X_train, y_train, X_test, y_test, batch_size=64, epochs=10, learning_rate=0.0001)
    print("Training complete.")

if __name__ == "__main__":
    main()