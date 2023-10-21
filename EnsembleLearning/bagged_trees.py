import numpy as np

from math import log2
from matplotlib import pyplot as plt
from TreeNode import *
from statistics import median


def predict(tree, example):
    if len(tree.children) == 0:
        return 1 if tree.name == "yes" else -1

    for child in tree.children:
        if child.edge_label == example[tree.name]:
            return predict(child, example)

    return 1 if tree.name == "yes" else -1


def predict_bagged_final(trees, example):
    predictions = []
    for t in trees:
        predictions.append(predict(t, example))
    return max(set(predictions), key=predictions.count)


def find_best_attribute(S, attributes):
    best_gain = -1
    best_attribute = next(iter(attributes))
    total = len(S[best_attribute])
    labels = set(S["y"])

    overall_label_proportions = []
    for l in labels:
        overall_label_proportions.append(S["y"].count(l) / len(S["y"]))

    label_gain = information_gain(overall_label_proportions)

    for a in attributes:
        values = set(S[a])
        gains = []
        for v in values:
            weight = S[a].count(v) / len(S[a])
            label_proportions = []
            subset_labels = []
            for i in range(len(S[a])):
                if S[a][i] == v:
                    subset_labels.append(S["y"][i])
            for l in labels:
                label_proportions.append(subset_labels.count(l) / total)

            gain = information_gain(label_proportions)
            gains.append(weight * gain)

        final_gain = label_gain - sum(gains)
        if final_gain > best_gain:
            best_gain = final_gain
            best_attribute = a

    return best_attribute


def information_gain(proportions):
    proportions = [p for p in proportions if p != 0]
    gain = 0
    for p in proportions:
        gain -= (p * log2(p))

    return gain


def find_median(S, attribute):
    return median(S[attribute])


def id3(S, attributes):
    if all(map(lambda x: x == S["y"][0], S["y"])):
        return TreeNode(S["y"][0])

    elif len(attributes) == 0:
        return TreeNode(max(set(S["y"]), key=S["y"].count))
    else:
        A = find_best_attribute(S, attributes)
        root = TreeNode(A)

        values = set(S[A])
        for value in values:
            s_v = {k: [x for i, x in enumerate(v) if S[A][i] == value]
                   for k, v in S.items()}
            if len(s_v) == 0:
                TreeNode(max(set(S["y"]), key=S["y"].count), parent=root, edge_label=value)
            else:
                subtree = id3(s_v, attributes - {A})
                subtree.edge_label = value
                root.add_child(subtree)

        return root


def main():
    # Setup train data
    data = {"age": [], "job": [], "marital": [], "education": [], "default": [], "balance": [], "housing": [],
            "loan": [],
            "contact": [], "day": [], "month": [], "duration": [], "campaign": [], "pdays": [],
            "previous": [], "poutcome": [], "y": []}

    with open('data/bank/train.csv') as f:
        for line in f:
            terms = line.strip().split(',')
            for i in range(len(terms)):
                attributes = list(data.keys())
                data[attributes[i]].append(int(terms[i]) if terms[i].lstrip('-').isnumeric() else terms[i])

    median_age = median(data['age'])
    data['age'] = [0 if d < median_age else 1 for d in data['age']]

    median_balance = median(data['balance'])
    data['balance'] = [0 if d < median_balance else 1 for d in data['balance']]

    median_day = median(data['day'])
    data['day'] = [0 if d < median_day else 1 for d in data['day']]

    median_duration = median(data['duration'])
    data['duration'] = [0 if d < median_duration else 1 for d in data['duration']]

    median_campaign = median(data['campaign'])
    data['campaign'] = [0 if d < median_campaign else 1 for d in data['campaign']]

    median_pdays = median(data['pdays'])
    data['pdays'] = [0 if d < median_pdays else 1 for d in data['pdays']]

    median_previous = median(data['previous'])
    data['previous'] = [0 if d < median_previous else 1 for d in data['previous']]

    # Setup Test Data
    test = {"age": [], "job": [], "marital": [], "education": [], "default": [], "balance": [], "housing": [],
            "loan": [], "contact": [], "day": [], "month": [], "duration": [], "campaign": [], "pdays": [],
            "previous": [], "poutcome": [], "y": []}

    with open('data/bank/test.csv') as f:
        for line in f:
            terms = line.strip().split(',')
            for i in range(len(terms)):
                attributes = list(test.keys())
                test[attributes[i]].append(int(terms[i]) if terms[i].lstrip('-').isnumeric() else terms[i])

    median_age = median(test['age'])
    test['age'] = [0 if d < median_age else 1 for d in test['age']]

    median_balance = median(test['balance'])
    test['balance'] = [0 if d < median_balance else 1 for d in test['balance']]

    median_day = median(test['day'])
    test['day'] = [0 if d < median_day else 1 for d in test['day']]

    median_duration = median(test['duration'])
    test['duration'] = [0 if d < median_duration else 1 for d in test['duration']]

    median_campaign = median(test['campaign'])
    test['campaign'] = [0 if d < median_campaign else 1 for d in test['campaign']]

    median_pdays = median(test['pdays'])
    test['pdays'] = [0 if d < median_pdays else 1 for d in test['pdays']]

    median_previous = median(test['previous'])
    test['previous'] = [0 if d < median_previous else 1 for d in test['previous']]

    T = 500
    m = len(data["y"])
    trees = []

    print("* Executing bagged training algorithm...")
    for t in range(T):
        print(f"Iteration {t}/{T}")
        # Generates list of indices using discrete uniform distribution
        random_indices = list(np.random.randint(low=0, high=m, size=m))
        bagged_data = {"age": [], "job": [], "marital": [], "education": [], "default": [], "balance": [],
                       "housing": [],
                       "loan": [],
                       "contact": [], "day": [], "month": [], "duration": [], "campaign": [], "pdays": [],
                       "previous": [], "poutcome": [], "y": []}
        for i in random_indices:
            for key, value in data.items():
                bagged_data[key].append(value[i])
        trees.append(id3(data, attributes={"age", "job", "marital", "education", "default",
                                           "balance", "housing", "loan", "contact", "day", "month",
                                           "duration", "campaign",
                                           "pdays", "previous", "poutcome"}))

    print("** Running predictions to evaluate on training set...")

    train_errors = []
    for t in range(T):
        print(f"Prediction {t}/{T}")
        trials = []
        for i in range(len(data["age"])):
            example = {}
            for key, value in data.items():
                example[key] = value[i]
            trials.append(predict_bagged_final(trees[0:t + 1], example) == (1 if example['y'] == "yes" else -1))
        train_errors.append(trials.count(False) / len(trials))

    print("*** Running predictions to evaluate on test set...")

    test_errors = []
    for t in range(T):
        print(f"Prediction {t}/{T}")
        trials = []
        for i in range(len(test["age"])):
            example = {}
            for key, value in test.items():
                example[key] = value[i]
            trials.append(predict_bagged_final(trees[0:t + 1], example) == (1 if example['y'] == "yes" else -1))
        test_errors.append(trials.count(False) / len(trials))

    print("**** Outputting result plots...")

    plt.plot([x for x in range(T)], train_errors, label="Training Error")
    plt.plot([x for x in range(T)], test_errors, label="Testing Error")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
