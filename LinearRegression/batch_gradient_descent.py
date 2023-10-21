import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Loading data into pandas this time for ease in vector calculations
    data = pd.read_csv("data/concrete/train.csv")
    test = pd.read_csv("data/concrete/test.csv")

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    x_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    tolerance_level = 1e-6
    r = 1.0
    w = np.zeros(x.shape[1])
    cost_values = []

    while True:
        cost_gradient = np.dot(x.T, x.dot(w) - y) / len(y)
        new_weight = w - r * cost_gradient

        weight_delta = np.linalg.norm(new_weight - w)

        cost = np.sum((x.dot(w) - y) ** 2) / (2 * len(y))
        cost_values.append(cost)

        # Convergence
        if weight_delta < tolerance_level:
            break

        w = new_weight
        r *= 0.5

    # Plot
    plt.plot(cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Cost Function Convergence')
    plt.show()

    # Print Values
    print("Learned Weight Vector:", w)
    print("Learning Rate:", r)
    test_cost = np.sum((x_test.dot(w) - y_test) ** 2) / (2 * len(y_test))
    print("Cost Function Value:", test_cost)


if __name__ == "__main__":
    main()
