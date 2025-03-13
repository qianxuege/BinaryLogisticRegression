import numpy as np
import argparse


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)

def dJ(theta, xi, yi):
    pred = sigmoid(np.dot(xi, theta))
    return (pred - yi) * xi

def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float
) -> None:    
    # Train using SGD
    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            gradient = dJ(theta, X[i], y[i])
            theta -= learning_rate * gradient
    return theta


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    predictions = sigmoid(X @ theta) >= 0.5
    return predictions


def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    return np.mean(y != y_pred)

def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the labels and the feature vectors.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='str')
    y = dataset[:, 0].astype(float).astype(int)  # First column as labels
    X = dataset[:, 1:].astype(float)  # Rest of the columns as features
    return y,X

def main():
    num_epoch = int(args.num_epoch)
    learning_rate = float(args.learning_rate)

    y_train, X_train = load_tsv_dataset(args.train_input)
    y_test, X_test = load_tsv_dataset(args.test_input)

    # fold intercept
    X_train_w_intercept = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_w_intercept = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # initialize weights
    theta = np.zeros(X_train_w_intercept.shape[1])
    new_theta = train(theta, X_train_w_intercept, y_train, num_epoch, learning_rate)

    # Predictions
    predicted_train_labels = predict(new_theta, X_train_w_intercept)
    predicted_test_labels = predict(new_theta, X_test_w_intercept)

    # Compute errors
    train_error = compute_error(y_train, predicted_train_labels)
    test_error = compute_error(y_test, predicted_test_labels)

    # Write predictions
    np.savetxt(args.train_out, predicted_train_labels.astype(int), fmt='%d')
    np.savetxt(args.test_out, predicted_test_labels.astype(int), fmt='%d')

    # Write metrics
    with open(args.metrics_out, 'w') as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, 
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()
    main()
