import numpy as np
import pandas as pd


def train(X_train, y_train, epochs=10, learning_rate=1):
    X = X_train
    y = y_train
    W = np.zeros(X.shape[1])
    m = 0
    counts = [0]
    weight_vectors = [np.copy(W)]
    r = learning_rate

    for _ in range(epochs):
        for i in range(len(y)):
            prediction = predict(W, X[i])
            if prediction != y[i]:
                if y[i] == 1:
                    W += r * X[i]
                else:
                    W -= r * X[i]
                m = m + 1
                weight_vectors.append(np.copy(W))
                counts.append(1)
            else:
                counts[m] += 1

    return weight_vectors, counts


def predict(weights, example):
    return 0 if weights.dot(example) < 0 else 1


def predict_from_votes(weight_vectors, counts, example):
    voted_result = sum([counts[i] * (-1 if predict(weight_vectors[i], example) == 0 else 1)
                        for i in range(len(counts))])
    return 0 if voted_result < 0 else 1


def main():
    # Data Setup
    data = pd.read_csv("data/bank-note/train.csv", header=None)
    test = pd.read_csv("data/bank-note/test.csv", header=None)

    X_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    # Train and calculate error on test data
    weight_vectors, counts = train(X_train, y_train)

    test_results = [predict_from_votes(weight_vectors, counts, X_test[i]) == y_test[i] for i in range(len(y_test))]
    final_error = test_results.count(False) / len(test_results)

    print(f"Distinct learned weights and counts are:")
    for i in range(len(weight_vectors)):
        print(f"\tWeights: {weight_vectors[i]}\tCount: {counts[i]}")

    print(f"Final error is approximately {final_error}.")


if __name__ == "__main__":
    main()
