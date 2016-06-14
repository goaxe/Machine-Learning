# -*- coding:utf-8 -*-
from numpy import *
import operator
import os


def classify0(inX, data_set, labels, k):
    data_set_size = data_set.shape[0]
    diff_mat = tile(inX, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5
    sorted_distances = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_distances[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_class_count = sorted(class_count.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def createDataSet():
    group = array([[1.0, 1.0], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    f = open(filename)
    lines = f.readlines()
    line_num = len(lines)
    ret_mat = zeros((line_num, 3))
    class_label_vector = []
    index = 0
    for line in lines:
        line = line.strip()
        splits = line.split('\t')
        ret_mat[index:] = splits[0:3]
        class_label_vector.append(int(splits[-1]))
        index += 1
    return ret_mat, class_label_vector


def autoNorm(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    # norm_data_set = zeros(shape(data_set))
    m = data_set.shape[0]
    norm_data_set = data_set - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals


def datingClassTest(filename):
    ho_ratio = 0.10
    dating_data_mat, dating_labels = file2matrix(filename)
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    m = norm_mat.shape[0]
    test_vecs_num = int(m * ho_ratio)
    error_count = 0
    for i in range(test_vecs_num):
        classifier_result = classify0(norm_mat[i, :],
                                      norm_mat[test_vecs_num:m, :],
                                      dating_labels[test_vecs_num:m], 3)
        print 'the classifier came back with: %d, the real answer is %d' % (
            classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print 'the total error rate is %f' % (error_count / float(test_vecs_num))


def classifyPerson(filename):
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(
        raw_input('percentage of time spend playing video games?'))
    ff_miles = float(raw_input('frequent flier miles earned per year?'))
    ice_cream = float(raw_input('liters of ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2matrix(filename)
    norm_mat, ranges, min_vals = autoNorm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr - min_vals) / ranges, norm_mat,
                                  dating_labels, 3)
    print 'you will probable like this person:', result_list[
        classifier_result - 1]


def img2vector(filename):
    ret_vec = zeros((1, 1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            ret_vec[0, 32 * i + j] = int(line[j])
    return ret_vec


def handwritingClassTest(training_file_dir, testing_file_dir):
    hw_labels = []

    training_file_list = os.listdir(training_file_dir)
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        filename = training_file_list[i].split('.')[0]
        class_num = int(filename.split('_')[0])
        hw_labels.append(class_num)
        training_mat[i, :] = img2vector(
            training_file_dir + '/' + training_file_list[i])

    test_file_list = os.listdir(testing_file_dir)
    error_count = 0.0
    test_count = len(test_file_list)
    for i in range(test_count):
        filename = test_file_list[i].split('.')[0]
        class_num = int(filename.split('_')[0])
        test_vec = img2vector(testing_file_dir + '/' + test_file_list[i])
        classifier_result = classify0(test_vec, training_mat, hw_labels, 3)
        if classifier_result != class_num:
            error_count += 1.0
    print 'total num of errors is %d, error rate is %f' % (
        error_count, error_count / float(test_count))


if __name__ == '__main__':
    # data_set, labels = createDataSet()
    # print classify0([0.9, 0.1], data_set, labels, 2)

    # filename = 'data/datingTestSet2.txt'
    # classifyPerson(filename)

    training_file_dir = 'digits/trainingDigits'
    testing_file_dir = 'digits/testDigits'

    handwritingClassTest(training_file_dir, testing_file_dir)

