from math import log2
from TreeNode import *


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
    labels = set(S["label"])

    overall_label_proportions = []
    for l in labels:
        overall_label_proportions.append(S["label"].count(l) / len(S["label"]))

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
                    subset_labels.append(S["label"][i])
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


def id3(S, attributes, max_depth, depth=-1, best_attribute_method="info_gain"):
    depth += 1
    if depth == max_depth:
        return TreeNode(max(set(S["label"]), key=S["label"].count))

    if all(map(lambda x: x == S["label"][0], S["label"])):
        if len(attributes) == 0:
            return TreeNode(max(set(S["label"]), key=S["label"].count))
        return TreeNode(S["label"][0])

    else:
        A = find_best_attribute(S, attributes, best_attribute_method)
        root = TreeNode(A)

        values = set(S[A])
        for value in values:
            s_v = {k: [x for i, x in enumerate(v) if S[A][i] == value]
                   for k, v in S.items()}
            if len(s_v) == 0:
                TreeNode(max(set(S["label"]), key=S["label"].count), parent=root, edge_label=value)
            else:
                subtree = id3(s_v, attributes - {A}, max_depth, depth=depth)
                subtree.edge_label = value
                root.add_child(subtree)

        return root


def main():
    data = {"buying": [], "maint": [], "doors": [], "persons": [], "lug_boot": [], "safety": [], "label": []}
    with open('data/car/train.csv') as f:
        for line in f:
            terms = line.strip().split(',')
            for i in range(len(terms)):
                attributes = list(data.keys())
                data[attributes[i]].append(terms[i])

    for method in ["info_gain", "gi", "me"]:
        for depth in range(1, 7):
            print("========================================================================")
            print(f"Method: {method}\nDepth: {depth}")
            tree = id3(data, max_depth=depth, attributes={"buying", "maint", "doors", "persons", "lug_boot", "safety"},
                       best_attribute_method=method)

            train_predictions = []

            for i in range(len(data["buying"])):
                example = {}
                for key, value in data.items():
                    example[key] = value[i]
                train_predictions.append(predict(tree, example) == example['label'])

            print(f"Training Error: {1 - (train_predictions.count(True) / len(train_predictions))}")

            test = {"buying": [], "maint": [], "doors": [], "persons": [], "lug_boot": [], "safety": [], "label": []}
            with open('data/car/test.csv') as f:
                for line in f:
                    terms = line.strip().split(',')
                    for i in range(len(terms)):
                        attributes = list(test.keys())
                        test[attributes[i]].append(terms[i])

            test_predictions = []

            for i in range(len(test["buying"])):
                example = {}
                for key, value in test.items():
                    example[key] = value[i]
                test_predictions.append(predict(tree, example) == example['label'])

            print(f"Test Error: {1 - (test_predictions.count(True) / len(test_predictions))}")


if __name__ == '__main__':
    main()
