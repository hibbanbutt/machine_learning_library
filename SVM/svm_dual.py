import numpy as np
import pandas as pd
from scipy.optimize import minimize


class DualSVM:
    def __init__(self, C):
        self.C = C
        self.a = None
        self.X = None
        self.y = None

    def objective(self, a):
        return 0.5 * np.sum(a * a * self.y * self.y * np.dot(self.X, self.X.T)) - np.sum(a)

    def constraint(self, a):
        return np.dot(a, self.y)

    def train(self, X, y):
        self.X = X
        self.y = y

        a_0 = np.zeros(len(X))
        bounds = [(0, self.C) for x in range(len(X))]

        self.a = minimize(self.objective, a_0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'eq', 'fun': lambda a: np.dot(a, self.y)},
                          options={'ftol': 1e-8, 'maxiter':10}).x

    def predict(self, X):
        return np.sign(np.sum(self.a * self.y * np.dot(X, self.X.T), axis=1))

    def test(self, X, y):
        return 1.0 - np.mean(self.predict(X) == y)

    def recover_weights_and_bias(self):
        support_vectors = self.a > 1e-5
        return np.sum(self.a * self.y * self.X.T, axis=1), np.mean(
            self.y[support_vectors] - np.sum(self.a * self.y * np.dot(self.X[support_vectors], self.X.T),
                                             axis=1))


def main():
    # Load data
    train_data = pd.read_csv("data/bank-note/train.csv")
    test_data = pd.read_csv("data/bank-note/test.csv")

    # Training data (with label normalizing into {-1, 1})
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_train[y_train == 0] = -1

    # Test data (with label normalizing into {-1, 1})
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    y_test[y_test == 0] = -1

    C_values = [100 / 873, 500 / 873, 700 / 873]

    for C in C_values:
        # Train
        model = DualSVM(C=C)
        model.train(X_train, y_train)

        # Test
        train_error = model.test(X_train, y_train)
        test_error = model.test(X_test, y_test)

        # Print Results for each C
        print("=========================")
        print(f"C: {C}")
        print("------------------")
        print(f"Training error: {train_error}")
        print(f"Test error: {test_error}")
        print(f"Weights: {model.recover_weights_and_bias()[0]}")


if __name__ == '__main__':
    main()
