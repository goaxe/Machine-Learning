# -*- coding:utf-8 -*-
import random

from numpy import *


def load_data_set(filename):
    data_mat = []
    label_mat = []
    f = open(filename)
    for line in f.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append((int(line_arr[2])))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))
    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights += alpha * data_matrix.transpose() * error
    return weights


def sto_grad_ascent(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * weights))
        error = class_labels[i] - h
        weights += alpha * error * array(data_matrix[i])
    return weights


def sto_grad_ascent1(data_matrix, class_labels, iter_num=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(iter_num):
        data_index = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights += alpha * error * array(data_matrix[rand_index])
            del (data_index[rand_index])
    return weights


def plot_best_fit(filename, weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set(filename)
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify_vector(in_vec, weights):
    prob = sigmoid(sum(in_vec * weights))
    if prob > 0.5:
        return 1.0
    return 0.0


def colic_test():
    f_train = open('data/horseColicTraining.txt')
    f_test = open('data/horseColicTest.txt')
    training_set = []
    trainging_labels = []
    for line in f_train.readlines():
        line = line.strip().split('\t')
        line_array = [float(line[i]) for i in range(21)]
        training_set.append(line_array)
        trainging_labels.append(float(line[21]))
    train_weights = sto_grad_ascent1(array(training_set), trainging_labels, 500)
    error_count = 0
    test_num = 0.0
    for line in f_test.readlines():
        test_num += 1.0
        line = line.strip().split('\t')
        line_array = [float(line[i]) for i in range(21)]
        if int(classify_vector(array(line_array), train_weights)) != int(
                line[21]):
            error_count += 1
    error_rate = float(error_count) / test_num
    print 'the error rate of this test is: %f' % error_rate
    return error_rate


def multi_test():
    test_num = 10
    error_sum = 0.0
    for k in range(test_num):
        error_sum += colic_test()
    print 'after %d iterations the average error rate is %f' % \
          (test_num, error_sum / float(test_num))


if __name__ == '__main__':
    # filename = 'data/testSet.txt'
    # data_mat, label_mat = load_data_set(filename)
    # weights = grad_ascent(data_mat, label_mat)
    # plot_best_fit(filename, weights)
    # weights = sto_grad_ascent(data_mat, label_mat)
    # plot_best_fit(filename, weights)
    # weights = sto_grad_ascent1(data_mat, label_mat)
    # plot_best_fit(filename, weights)

    multi_test()
