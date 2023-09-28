from math import log2
from TreeNode import *
from statistics import median


def predict(tree, example):
    if len(tree.children) == 0:
        return tree.name

    for child in tree.children:
        if child.edge_label == example[tree.name]:
            return predict(child, example)

    return tree.name


def find_best_attribute(S, attributes, method):
    best_gain = -1
    best_attribute = next(iter(attributes))
    total = len(S[best_attribute])
    labels = set(S["y"])

    overall_label_proportions = []
    for l in labels:
        overall_label_proportions.append(S["y"].count(l) / len(S["y"]))

    if method == "gi":
        label_gain = gini_index(overall_label_proportions)
    elif method == "me":
        label_gain = majority_error(overall_label_proportions)
    else:
        label_gain = information_gain(overall_label_proportions)

    for a in attributes:
        values = set(S[a])
        gains = []
        for v in values:
            weight = S[a].count(v)
            label_proportions = []
            subset_labels = []
            for i in range(len(S[a])):
                if S[a][i] == v:
                    subset_labels.append(S["y"][i])
            for l in labels:
                label_proportions.append(subset_labels.count(l) / total)
            if method == "gi":
                gain = gini_index(label_proportions)
            elif method == "me":
                gain = majority_error(label_proportions)
            else:
                gain = information_gain(label_proportions)

            gains.append(weight * gain)

        final_gain = label_gain - sum(gains)
        if final_gain > best_gain:
            best_gain = final_gain
            best_attribute = a

    return best_attribute


def gini_index(proportions):
    proportions = [p for p in proportions if p != 0]
    gini = 0
    for p in proportions:
        gini += p ** 2

    return 1 - gini


def majority_error(proportions):
    proportions = [p for p in proportions if p != 0]
    return min(proportions)


def information_gain(proportions):
    proportions = [p for p in proportions if p != 0]
    gain = 0
    for p in proportions:
        gain -= log2(p)

    return gain


def find_median(S, attribute):
    return median(S[attribute])


def id3(S, attributes, max_depth, depth=-1, best_attribute_method="info_gain"):
    depth += 1

    if depth == max_depth:
        return TreeNode(max(set(S["y"]), key=S["y"].count))

    if all(map(lambda x: x == S["y"][0], S["y"])):
        if len(attributes) == 0:
            return TreeNode(max(set(S["y"]), key=S["y"].count))
        return TreeNode(S["y"][0])

    else:
        A = find_best_attribute(S, attributes, best_attribute_method)
        root = TreeNode(A)

        values = set(S[A])
        for value in values:
            s_v = {k: [x for i, x in enumerate(v) if S[A][i] == value]
                   for k, v in S.items()}
            if len(s_v) == 0:
                TreeNode(max(set(S["y"]), key=S["y"].count), parent=root, edge_label=value)
            else:
                subtree = id3(s_v, attributes - {A}, max_depth, depth=depth)
                subtree.edge_label = value
                root.add_child(subtree)

        return root


def main():
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

    # for key, value in data.items():
    #     for i in range(len(value)):
    #         if value[i] == "unknown":
    #             data[key][i] = max(set(value), key=value.count)

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

    for method in ["info_gain", "gi", "me"]:
        for depth in range(1, 17):
            print("========================================================================")
            print(f"Method: {method} \nDepth: {depth}")
            tree = id3(data, max_depth=depth, attributes={"age", "job", "marital", "education", "default",
                                                          "balance", "housing", "loan", "contact", "day", "month",
                                                          "duration", "campaign",
                                                          "pdays", "previous", "poutcome", "y"},
                       best_attribute_method=method)

            train_predictions = []

            for i in range(len(data["age"])):
                example = {}
                for key, value in data.items():
                    example[key] = value[i]
                train_predictions.append(predict(tree, example) == example['y'])

            print(f"Training Error: {1 - (train_predictions.count(True) / len(train_predictions))}")

            test = {"age": [], "job": [], "marital": [], "education": [], "default": [], "balance": [], "housing": [],
                    "loan": [], "contact": [], "day": [], "month": [], "duration": [], "campaign": [], "pdays": [],
                    "previous": [], "poutcome": [], "y": []}

            with open('data/bank/test.csv') as f:
                for line in f:
                    terms = line.strip().split(',')
                    for i in range(len(terms)):
                        attributes = list(test.keys())
                        test[attributes[i]].append(int(terms[i]) if terms[i].lstrip('-').isnumeric() else terms[i])

            #            for key, value in data.items():
            #                 for i in range(len(value)):
            #                     if value[i] == "unknown":
            #                         data[key][i] = max(set(value), key=value.count)

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

            test_predictions = []

            for i in range(len(test["age"])):
                example = {}
                for key, value in test.items():
                    example[key] = value[i]
                # prediction = predict(tree, example)
                test_predictions.append(predict(tree, example) == example['y'])

                # print(prediction)

            print(f"Test Error: {1 - (test_predictions.count(True) / len(test_predictions))}")


if __name__ == '__main__':
    main()
