import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class PrimalSVM:
    def __init__(self, C, learning_rate_0, a, T, schedule):
        self.C = C
        self.a = a
        self.T = T
        self.w = None
        self.b = 0
        self.objective_curve = []
        self.schedule = schedule
        self.learning_rate = learning_rate_0
        self.learning_rate_0 = learning_rate_0

    def loss(self, X, y):
        return np.maximum(0, 1 - y * (np.dot(X, self.w) + self.b))

    def gradient(self, X, y):
        return (np.zeros_like(self.w), 0) if y * (np.dot(X, self.w) + self.b) >= 1 else (-y * X, -y)

    def objective(self, X, y):
        return 0.5 * np.dot(self.w, self.w) + self.C * np.sum(self.loss(X, y))

    def train(self, X, y):
        self.w = np.zeros(X.shape[1])

        for t in range(self.T):
            X, y = shuffle(X, y)

            # Schedule learning rate calculation + SSGD
            for i in range(len(X)):
                if self.schedule == 'a':
                    self.learning_rate = self.learning_rate_0 / (1 + (self.learning_rate_0 / self.a) * (t * len(X) + i))
                elif self.schedule == 'b':
                    self.learning_rate = self.learning_rate_0 / (1 + t)
                else:
                    print("Invalid schedule option provided. Aborting training...")
                    return

                gradient_w, gradient_b = self.gradient(X[i], y[i])
                self.w = self.w - self.learning_rate * (gradient_w + 2 / ((t + 1) * self.C) * self.w)
                self.b = self.b - self.learning_rate * gradient_b

            # Call objective function to track curve for convergence diagnosis
            current_objective = self.objective(X, y)
            self.objective_curve.append(current_objective)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

    def test(self, X, y):
        return 1.0 - np.mean(self.predict(X) == y)


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
        model = PrimalSVM(C=C, learning_rate_0=0.01, a=100, T=100, schedule='a')
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
        print(f"Final Learning Rate: {model.learning_rate}")
        print(f"Weights: {model.w}")


if __name__ == '__main__':
    main()
