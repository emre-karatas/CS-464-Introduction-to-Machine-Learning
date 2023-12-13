"""
Emre KARATAŞ
22001641
CS 464-Section 01
"""
import matplotlib.pyplot as plt
import numpy as np
import gzip
import seaborn as sns
import math


def one_hot_encoding(labels):
    """
    Encode labels using one-hot encoding.

    :param labels: A numpy array of shape (n_samples,) containing the labels.
    :return: A numpy array of shape (n_samples, n_classes) where n_samples is the number of labels, and n_classes is the number of unique classes.
    """
    n_labels = labels.shape[0]
    n_classes = 10
    one_hot = np.zeros((n_labels, n_classes))
    one_hot[np.arange(n_labels), labels] = 1
    return one_hot


def read_pixels(data_path):
    """
    :param data_path: The path to the data file.
    :return: The flattened pixels as a 2D array.

    This method reads pixel data from the specified file and returns the flattened pixels as a 2D array. The pixel data is assumed to be stored in a binary gzip file format.

    Example usage:
        data_path = "path/to/data.gz"
        pixels = read_pixels(data_path)
    """
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    flattened_pixels = normalized_pixels.reshape(-1, 28 * 28)
    return flattened_pixels


def read_labels(data_path):
    """
    :param str data_path: The path to the gzipped label data file.
    :return: A NumPy array containing one-hot encoded labels.
    :rtype: numpy.ndarray

    """
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels


def read_dataset():
    """
    Reads and loads a dataset for training and testing.

    :return: A tuple containing the following:
        - X_train: Training data, an array of pixel values for each image.
        - y_train: Training labels, an array of labels corresponding to each image.
        - X_test: Testing data, an array of pixel values for each image.
        - y_test: Testing labels, an array of labels corresponding to each image.
    """
    X_train = read_pixels("data/train-images-idx3-ubyte.gz")
    y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    X_test = read_pixels("data/t10k-images-idx3-ubyte.gz")
    y_test = read_labels("data/t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test


# Read and split the dataset
X_train, y_train, X_test, y_test = read_dataset()
X_train, X_val = X_train[:50000], X_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

# Adding bias term to the datasets
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


def softmax(z):
    """

    Calculate the softmax function over the input values.

    :param z: A numpy array representing the input values.
    :return: A numpy array containing the softmax values of the input array.

    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def initialize_weights(dim, init_type='normal'):
    """
    Initializes weights for a neural network layer.

    :param dim: The dimension of the weight matrix.
    :param init_type: (Optional) The type of initialization to be used. Default is 'normal'.
                      It can be 'zero', 'uniform' or 'normal'.
    :return: The initialized weight matrix of shape (dim, 10).

    :rtype: numpy.ndarray
    """
    if init_type == 'zero':
        return np.zeros((dim, 10))
    elif init_type == 'uniform':
        return np.random.uniform(-1, 1, (dim, 10))
    elif init_type == 'normal':
        return np.random.normal(0, 1, (dim, 10))


def compute_loss(y, y_hat, weights, lambda_reg):
    """
    Compute the loss for a given set of predictions.

    :param y: The true labels in one-hot encoded format.
    :param y_hat: The predicted probabilities for each class.
    :param weights: The weights used for regularization.
    :param lambda_reg: The regularization parameter.

    :return: The computed loss.

    """
    cross_entropy_loss = -np.mean(np.sum(y * np.log(y_hat), axis=1))
    l2_loss = (lambda_reg / 2) * np.sum(weights ** 2)
    return cross_entropy_loss + l2_loss


def train(X, y, epochs, batch_size, learning_rate, lambda_reg, init_type='normal'):
    """
    Train the model using gradient descent.

    :param X: The input features.
    :param y: The target labels.
    :param epochs: The number of training epochs.
    :param batch_size: The batch size for mini-batch gradient descent.
    :param learning_rate: The learning rate for gradient descent.
    :param lambda_reg: The regularization parameter.
    :param init_type: The initialization type for weights. Default is 'normal'.
    :return: The trained weights.

    """
    weights = initialize_weights(X.shape[1], init_type)
    for epoch in range(epochs):
        for i in range(0, X.shape[0], batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            y_hat = softmax(np.dot(X_batch, weights))
            loss = compute_loss(y_batch, y_hat, weights, lambda_reg)
            gradients = np.dot(X_batch.T, (y_hat - y_batch))
            gradients += lambda_reg * weights  # Apply regularization
            weights -= learning_rate * gradients
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')
    return weights


# Hyperparameters
epochs = 100
batch_size = 200
learning_rate = 5e-4
lambda_reg = 1e-4

# Training the model
weights = train(X_train, y_train, epochs, batch_size, learning_rate, lambda_reg)


# Function to predict labels
def predict(X, weights):
    """
    Predicts the class labels using the softmax function.

    :param X: Input features in the shape of (n_samples, n_features).
    :param weights: Coefficients of the linear model in the shape of (n_features, n_classes).

    :return: Predicted class labels for each sample in the shape of (n_samples,).
    """
    return softmax(np.dot(X, weights)).argmax(axis=1)


# Predict on test data
y_pred = predict(X_test, weights)

# Calculate test accuracy
accuracy = np.mean(y_pred == y_test.argmax(axis=1))
print(f"Test Accuracy: {accuracy * 100:.2f}%")


def compute_confusion_matrix(true, pred, num_classes):
    """
    Compute the confusion matrix for a classification model.

    :param true: The true labels of the data samples.
    :type true: list or numpy array

    :param pred: The predicted labels of the data samples.
    :type pred: list or numpy array

    :param num_classes: The number of unique classes in the dataset.
    :type num_classes: int

    :return: The confusion matrix, a square numpy array of shape (num_classes, num_classes),
             where each element represents the count of samples classified in each class.
    :rtype: numpy array
    """
    matrix = np.zeros((num_classes, num_classes))
    for t, p in zip(true, pred):
        matrix[t, p] += 1
    return matrix


# Generate confusion matrix
num_classes = 10  # For MNIST data
confusion_mtx = compute_confusion_matrix(y_test.argmax(axis=1), y_pred, num_classes)

# Plotting the confusion matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='viridis', cbar=False, linewidths=.5)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Heatmap')
plt.show()


def test_batch_sizes(X_train, y_train, X_test, y_test, batch_sizes, epochs):
    for batch_size in batch_sizes:
        accuracies = []
        print(f"Testing with batch size: {batch_size}")
        for epoch in range(0, epochs + 1, 10):  # Increment every 10 epochs
            weights = train(X_train, y_train, epoch, batch_size, learning_rate, lambda_reg, init_type='normal')
            y_pred = predict(X_test, weights)
            accuracy = np.mean(y_pred == y_test.argmax(axis=1))
            accuracies.append(accuracy)
            print(f'Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%')
        plt.plot(range(0, epochs + 1, 10), accuracies, label=f'Batch Size: {batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance for Different Batch Sizes')
    plt.legend()
    plt.show()


def test_init_methods(X_train, y_train, X_test, y_test, init_methods, epochs):
    """
    Tests multiple initialization methods for training a model. It takes in the training and testing data, a list of initialization methods, and the number of epochs. For each
    initialization method, it trains the model with the specified init_method using the train() method, predicts the output labels using the predict() method, calculates the accuracy,
    and stores the accuracy values for each epoch. Finally, it plots the accuracy values for each initialization method over the epochs and displays the plot.
    """
    for init_method in init_methods:
        accuracies = []
        print(f"Testing with init method: {init_method}")
        for epoch in range(0, epochs + 1, 10):  # Increment every 10 epochs
            weights = train(X_train, y_train, epoch, batch_size, learning_rate, lambda_reg, init_type=init_method)
            y_pred = predict(X_test, weights)
            accuracy = np.mean(y_pred == y_test.argmax(axis=1))
            accuracies.append(accuracy)
            print(f'Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%')
        plt.plot(range(0, epochs + 1, 10), accuracies, label=f'Init Method: {init_method}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance for Different Initialization Methods')
    plt.legend()
    plt.show()


def test_learning_rates(X_train, y_train, X_test, y_test, learning_rates, epochs):
    """
    Tests the performance of a model for different learning rates. Trains the model using the training data
    and labels with each specified learning rate, evaluates the model's accuracy on the testing data and
    labels for every 10th epoch, and plots the accuracy over the number of epochs for each learning rate.
    """
    for lr in learning_rates:
        accuracies = []
        print(f"Testing with learning rate: {lr}")
        for epoch in range(0, epochs + 1, 10):  # Increment every 10 epochs
            weights = train(X_train, y_train, epoch, batch_size, lr, lambda_reg, init_type='normal')
            y_pred = predict(X_test, weights)
            accuracy = np.mean(y_pred == y_test.argmax(axis=1))
            accuracies.append(accuracy)
            print(f'Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%')
        plt.plot(range(0, epochs + 1, 10), accuracies, label=f'Learning Rate: {lr}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance for Different Learning Rates')
    plt.legend()
    plt.show()


def test_lambda_regs(X_train, y_train, X_test, y_test, lambda_regs, epochs):
    """
    Tests the performance of a model with different regularization coefficients. It plots the model accuracy over the specified number of epochs for each regularization coefficient.
    """
    for reg in lambda_regs:
        accuracies = []
        print(f"Testing with lambda reg: {reg}")
        for epoch in range(0, epochs + 1, 10):  # Increment every 10 epochs
            weights = train(X_train, y_train, epoch, batch_size, learning_rate, reg, init_type='normal')
            y_pred = predict(X_test, weights)
            accuracy = np.mean(y_pred == y_test.argmax(axis=1))
            accuracies.append(accuracy)
            print(f'Epoch {epoch}, Accuracy: {accuracy * 100:.2f}%')
        plt.plot(range(0, epochs + 1, 10), accuracies, label=f'Lambda Reg: {reg}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Performance for Different Regularization Coefficients')
    plt.legend()
    plt.show()


def hyperparameter_testing(X_train, y_train, X_test, y_test, epochs):
    """
    Hyperparameter testing for a machine learning model.

    :param X_train: The training data.
    :type X_train: numpy.ndarray
    :param y_train: The training labels.
    :type y_train: numpy.ndarray
    :param X_test: The test data.
    :type X_test: numpy.ndarray
    :param y_test: The test labels.
    :type y_test: numpy.ndarray
    :param epochs: The number of epochs to train for.
    :type epochs: int
    :return: None
    :rtype: None
    """
    batch_sizes = [1, 64, 50000]
    init_methods = ['zero', 'uniform', 'normal']
    learning_rates = [0.1, 1e-3, 1e-4, 1e-5]
    lambda_regs = [1e-2, 1e-4, 1e-9]

    print("Testing different batch sizes")
    test_batch_sizes(X_train, y_train, X_test, y_test, batch_sizes, epochs)

    print("\nTesting different weight initialization methods")
    test_init_methods(X_train, y_train, X_test, y_test, init_methods, epochs)

    print("\nTesting different learning rates")
    test_learning_rates(X_train, y_train, X_test, y_test, learning_rates, epochs)

    print("\nTesting different regularization coefficients")
    test_lambda_regs(X_train, y_train, X_test, y_test, lambda_regs, epochs)


hyperparameter_testing(X_train, y_train, X_test, y_test, 100)


def evaluate_optimal_model(X_train, y_train, X_test, y_test, best_hyperparams, epochs):
    """
    Retrain the model with the best hyperparameters, evaluate on the test set,
    and display the test accuracy and confusion matrix.

    :param X_train: Training data
    :param y_train: Training labels
    :param X_test: Test data
    :param y_test: Test labels
    :param best_hyperparams: Dictionary of best hyperparameters
    :param epochs: Number of epochs to train
    """
    # Retrain the model with the best hyperparameters
    weights = train(X_train, y_train, epochs, best_hyperparams['batch_size'],
                    best_hyperparams['learning_rate'], best_hyperparams['lambda_reg'],
                    best_hyperparams['init_method'])

    # Evaluate the model on the test set
    y_pred = predict(X_test, weights)
    test_accuracy = np.mean(y_pred == y_test.argmax(axis=1))
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Display the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', cbar=False, linewidths=.5)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix Heatmap for Best Model')
    plt.show()
    visualize_weights(weights, img_shape=(28, 28))
    metrics = calculate_metrics(confusion_mtx)
    for class_id, scores in metrics.items():
        print(f"{class_id}: {scores}")


def visualize_weights(weights, img_shape=None):
    """
    Visualize the weight vectors as images.

    :param weights: A NumPy array of weights of shape (n_features, n_classes)
    :param img_shape: Tuple specifying the shape of the images. If None, it will be inferred.
    """
    num_classes = weights.shape[1]
    weight_length = weights.shape[0]

    # Check for bias term and adjust weight_length
    if weights.shape[0] - 1 == img_shape[0] * img_shape[1]:
        weight_length -= 1  # Exclude the bias term
        weights = weights[1:, :]

    if img_shape is None:
        side_length = int(math.sqrt(weight_length))
        img_shape = (side_length, side_length)

    # Calculate the number of rows and columns for the subplot
    n_cols = min(num_classes, 5)
    n_rows = math.ceil(num_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3))
    axes = axes.flatten()

    for i in range(num_classes):
        ax = axes[i]
        img = weights[:, i].reshape(img_shape)
        im = ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Class {i}')

    fig.colorbar(im, ax=axes[-1], orientation='vertical')

    plt.tight_layout()
    plt.show()


def calculate_metrics(confusion_mtx):
    metrics = {}
    num_classes = confusion_mtx.shape[0]

    for i in range(num_classes):
        TP = confusion_mtx[i, i]
        FP = confusion_mtx[:, i].sum() - TP
        FN = confusion_mtx[i, :].sum() - TP
        TN = confusion_mtx.sum() - (TP + FP + FN)

        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0

        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        f2_score = (1 + 2 ** 2) * precision * recall / (2 ** 2 * precision + recall) if (
                                                                                                    2 ** 2 * precision + recall) != 0 else 0

        metrics[f'Class {i}'] = {'Precision': precision, 'Recall': recall, 'F1 Score': f1_score, 'F2 Score': f2_score}

    return metrics


best_hyperparams = {
    'batch_size': 64, 'learning_rate': 1e-3,
    'lambda_reg': 1e-2, 'init_method': 'zero'
}
evaluate_optimal_model(X_train, y_train, X_test, y_test, best_hyperparams, 100)
