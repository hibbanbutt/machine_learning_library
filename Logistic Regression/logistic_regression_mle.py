import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.utils import shuffle


class LogisticRegression:
    def __init__(self, input_size):
        self.weights = np.zeros((input_size, 1))

    def sigmoid(self, z):
        return expit(z)

    def gradient(self, X, y):
        predictions = self.sigmoid(np.dot(X, self.weights))
        error = predictions - y.flatten()
        gradient = (1 / len(y)) * np.dot(X.T, error)
        return gradient

    def train(self, X_train, y_train, epochs, initial_learning_rate, d):
        for t in range(epochs):
            X, y = shuffle(X_train, y_train)

            learning_rate = initial_learning_rate / (1 + (initial_learning_rate / d) * t)
            for i in range(len(y_train)):
                xi = X[i, :].reshape(1, -1)
                yi = y[i]
                gradient = self.gradient(xi, yi)
                self.weights = self.weights - learning_rate * gradient.reshape(-1, 1)

    def test(self, X, y):
        test_predictions = self.predict(X)
        test_predictions[test_predictions == 0] = -1
        test_error = np.mean(np.sign(test_predictions) != y.flatten())
        return test_error

    def predict(self, X):
        return (self.sigmoid(np.dot(X, self.weights)) >= 0.5).astype(int)

    def loss(self, X, y):
        predictions = self.sigmoid(np.dot(X, self.weights))
        likelihood_term = -(1 / len(y)) * np.sum(
            y.flatten() * np.log(predictions) + (1 - y.flatten()) * np.log(1 - predictions))
        return likelihood_term


def main():
    # Load data
    train_data = pd.read_csv("../Logistic Regression/data/bank-note/train.csv")
    test_data = pd.read_csv("../Logistic Regression/data/bank-note/test.csv")

    # Training data (with label normalizing into {-1, 1})
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_train[y_train == 0] = -1
    y_train = np.expand_dims(y_train, axis=1)

    # Test data (with label normalizing into {-1, 1})
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    y_test[y_test == 0] = -1
    y_test = np.expand_dims(y_test, axis=1)


    initial_learning_rate = 0.1
    d = 100

    # Train
    model = LogisticRegression(X_train.shape[1])
    model.train(X_train, y_train, 100, initial_learning_rate, d)

    # Test
    train_error = model.test(X_train, y_train)
    test_error = model.test(X_test, y_test)

    # Print Results
    print("=========================")
    print(f"Training error: {train_error}")
    print(f"Test error: {test_error}")

if __name__ == "__main__":
    main()
