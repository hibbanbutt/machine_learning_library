import numpy as np
import pandas as pd

from sklearn.utils import shuffle


def train(X_train, y_train, epochs=10, learning_rate=1):
    X = X_train
    y = y_train
    W = np.zeros(X.shape[1])
    r = learning_rate

    for _ in range(epochs):
        X, y = shuffle(X, y, random_state=0)
        for i in range(len(y)):
            prediction = predict(W, X[i])
            if prediction != y[i]:
                if y[i] == 1:
                    W += r * X[i]
                else:
                    W -= r * X[i]

    return W


def predict(weights, example):
    return 0 if weights.dot(example) < 0 else 1


def main():
    # Data Setup
    data = pd.read_csv("data/bank-note/train.csv", header=None)
    test = pd.read_csv("data/bank-note/test.csv", header=None)

    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Train and calculate error on test data
    weights = train(X_train, y_train)
    test_results = [predict(weights, X_test[i]) == y_test[i] for i in range(len(y_test))]
    final_error = test_results.count(False) / len(test_results)

    print(f"Learned weights are {weights}.")
    print(f"Final error is approximately {final_error}.")


if __name__ == "__main__":
    main()
