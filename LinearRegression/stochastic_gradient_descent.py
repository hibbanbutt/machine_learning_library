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
    r = 0.1
    w = np.zeros(x.shape[1])
    cost_values = []
    iterations = 0

    while True:
        for i in range(len(y)):
            xi = x[i, :]
            yi = y[i]

            stoch_gradient = xi * (xi.dot(w) - yi)
            w = w - r * stoch_gradient

            iterations += 1

            cost = np.sum((x.dot(w) - y) ** 2) / (2 * len(y))
            cost_values.append(cost)

            # Convergence
            if iterations > 1 and abs(cost_values[-1] - cost_values[-2]) < tolerance_level:
                break

        r *= 0.99

        # Convergence again
        if iterations > 1 and abs(cost_values[-1] - cost_values[-2]) < tolerance_level:
            break

    # Plot
    plt.plot(cost_values)
    plt.xlabel('Number of Updates')
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
