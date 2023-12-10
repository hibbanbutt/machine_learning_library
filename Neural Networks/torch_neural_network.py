from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def init_tanh(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

def init_relu(layer):
    if type(layer) == nn.Linear:
        nn.init.kaiming_normal_(layer.weight)
        nn.init.zeros_(layer.bias)


def main():
    # Load data
    train_data = pd.read_csv("../Neural Networks/data/bank-note/train.csv")
    test_data = pd.read_csv("../Neural Networks/data/bank-note/test.csv")

    # Training data (with label normalizing into {-1, 1})
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    y_train[y_train == 0] = -1

    # Test data (with label normalizing into {-1, 1})
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values
    y_test[y_test == 0] = -1

    # Convert all data to Torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)


    dataset = TensorDataset(X_train, y_train)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    activations = ['relu', 'tanh']
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]

    for a in activations:
        print(f"ACTVATION: {a}")
        for d in depths:
            for w in widths:
                activation = nn.ReLU() if a == 'relu' else nn.Tanh()
                architecture = OrderedDict([('layer1', nn.Linear(X_train.shape[1], w)),
                                            (f"{a}1", activation)])

                for i in range(2, d):
                    architecture[f"layer{i}"] = nn.Linear(w, w)
                    architecture[f"{a}{i}"] = activation

                architecture[f"layer{d}"] = nn.Linear(w, 1)


                model = nn.Sequential(architecture)
                model.apply(init_relu if a == 'relu' else init_tanh)


                #Train
                optimizer = optim.Adam(model.parameters())
                num_epochs = 100
                for epoch in range(num_epochs):
                    for examples, labels in dataloader:
                        predictions = model(examples)
                        loss = nn.functional.mse_loss(predictions, labels.unsqueeze(1))

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()


                y_preds = model(X_train)


                # Print Results for each C
                print("=========================")
                print(f"Depth: {d}")
                print(f"Width: {w}")
                print("------------------")
                print(f"Training error: {1.0 - torch.mean(torch.eq(torch.sign(y_preds), y_train).float()).item()}")

                y_preds = model(X_test)

                print(f"Test error: {1.0 - torch.mean(torch.eq(torch.sign(y_preds), y_test).float()).item()}")



if __name__ == '__main__':
    main()
