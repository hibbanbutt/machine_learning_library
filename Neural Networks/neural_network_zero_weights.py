import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.special import expit


class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None

    def sigmoid(self, x):
        return expit(x)

    def sigmoid_prime(self, x):
        return expit(x)*(1-expit(x))

    def forward(self, input):
        self.input = input
        self.output = self.sigmoid(self.input)
        return self.output

    def backward(self, dL_dY):
        return self.sigmoid_prime(self.input) * dL_dY

class NeuralLayer:
    def __init__(self, input_size, output_size, random_weights = True):
        self.weights = np.random.normal(size=(input_size, output_size)) if random_weights else np.zeros(shape=(input_size, output_size))
        self.bias = np.random.normal(size=(1, output_size)) if random_weights else np.zeros(shape=(1, output_size))
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, dL_dY, learning_rate):
        dL_dX = np.dot(dL_dY, self.weights.T)
        dL_dW = np.dot(self.input.T, dL_dY)

        self.weights -= learning_rate * dL_dW
        self.bias -= learning_rate * dL_dY

        return dL_dX

    def static_backward(self, dL_dY):
        dL_dX = dL_dY.dot(self.weights.T)
        dL_dW = self.input.T.dot(dL_dY)

        return dL_dX, dL_dW

class NeuralNetwork:
    def __init__(self, hidden_layer_width, input_size, random_weights=True):
        self.layers = [NeuralLayer(input_size, hidden_layer_width, random_weights), Sigmoid(),
                       NeuralLayer(hidden_layer_width, hidden_layer_width, random_weights), Sigmoid(),
                       NeuralLayer(hidden_layer_width, 1, random_weights)]

    def predict(self, X):
        predictions = []

        for x in X:
            prediction = x
            for layer in self.layers:
                prediction = layer.forward(prediction)
            predictions.append(prediction)

        return predictions

    def train(self, X_train, y_train, epochs, initial_learning_rate, d):
        for t in range(epochs):
            learning_rate = initial_learning_rate / (1 + (initial_learning_rate * t / d))
            X, y = shuffle(X_train, y_train)

            for i in range(len(X)):
               # First fit the network to the example at hand
                prediction = X[i]
                for layer in self.layers:
                    prediction = layer.forward(prediction)
                loss_gradient = 2*(prediction - y[i])/y[i].size

                # Now propagate... backwards
                for layer in reversed(self.layers):
                    if type(layer) == NeuralLayer:
                        loss_gradient = layer.backward(loss_gradient, learning_rate)
                    else:
                        loss_gradient = layer.backward(loss_gradient)

    def test(self, X, y):
        return 1.0 - np.mean(np.sign(self.predict(X)) == y)


    def static_backpropagation(self, example, label):
        # All weights differentiated.
        # The first element in dW is a matrix of all the weight derivatives for the first layer's weights, second element
        # for the second layer's weights, etc.
        dW = []

        # First fit the network to the example at hand
        prediction = example
        for layer in self.layers:
            prediction = layer.forward(prediction)

        # Now backpropagate (without any learning... for HW purposes *see question 2a*)
        loss_gradient = self.loss_prime(label, prediction)
        for layer in reversed(self.layers):
            if type(layer) == NeuralLayer:
                loss_gradient, weights_gradient = layer.static_backward(loss_gradient)
                dW.append(weights_gradient)
            else:
                loss_gradient = layer.backward(loss_gradient)

        return reversed(dW)




def main():
    # Load data
    train_data = pd.read_csv("../Neural Networks/data/bank-note/train.csv")
    test_data = pd.read_csv("../Neural Networks/data/bank-note/test.csv")

    # Training data (with label normalizing into {-1, 1})
    X_train = train_data.iloc[:, :-1].values
    X_train = np.expand_dims(X_train, axis=1)
    y_train = train_data.iloc[:, -1].values
    y_train[y_train == 0] = -1
    y_train = np.expand_dims(y_train, axis=(1, 2))


    # Test data (with label normalizing into {-1, 1})
    X_test = test_data.iloc[:, :-1].values
    X_test = np.expand_dims(X_test, axis=1)
    y_test = test_data.iloc[:, -1].values
    y_test[y_test == 0] = -1
    y_test = np.expand_dims(y_train, axis=(1, 2))


    widths = [5, 10, 25, 50, 100]

    for w in widths:
        # Train
        model = NeuralNetwork(w, 4, random_weights=False)
        model.train(X_train, y_train, 100, 0.001, 100)

        # Test
        train_error = model.test(X_train, y_train)
        test_error = model.test(X_test, y_test)

        # Print Results for each width
        print("=========================")
        print(f"Width: {w}")
        print("------------------")
        print(f"Training error: {train_error}")
        print(f"Test error: {test_error}")



if __name__ == '__main__':
    main()
