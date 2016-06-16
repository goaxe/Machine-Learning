# -*- coding:utf-8 -*-

import treePloter
import operator
from math import log


def calcShannonEnt(data_set):
    num_entries = len(data_set)
    label_counts = {}
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


def createDataSet(filename):
    f = open(filename)
    lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return lenses, lenses_labels


def createDataSet1():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    data_set[0][-1] = 'maybe'
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def splitDataSet(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def chooseBestFeatureToSplit(data_set):
    features_num = len(data_set[0]) - 1
    base_entropy = calcShannonEnt(data_set)
    best_info_gain = 0.0
    best_feature = -1
    for i in range(features_num):
        feat_list = [example[i] for example in data_set]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data_set = splitDataSet(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calcShannonEnt(sub_data_set)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def majorityCnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def createTree(data_set, labels):
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) == 1:
        return majorityCnt(class_list)

    best_feat = chooseBestFeatureToSplit(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_vals = [example[best_feat] for example in data_set]
    unique_vals = set(feat_vals)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = createTree(
            splitDataSet(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    first_str = input_tree.keys()[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def storeTree(input_tree, filename):
    import pickle
    f = open(filename, 'w')
    pickle.dump(input_tree, f)
    f.close()


def grabTree(filename):
    import pickle
    f = open(filename, 'r')
    return pickle.load(f)


if __name__ == '__main__':
    # data_set, labels = createDataSet()
    # input_tree = createTree(data_set, labels[:])
    # print classify(input_tree, labels, [1, 0])
    # print classify(input_tree, labels, [1, 1])
    # filename = 'data/tree.txt'
    # storeTree(input_tree, filename)
    # print grabTree(filename)

    filename = 'data/lenses.txt'
    lenses, lenses_labels = createDataSet(filename)
    lenses_tree = createTree(lenses, lenses_labels)
    treePloter.createPlot(lenses_tree)

