# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation, update_parameters, \
    predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x, train_y, test_x, test_y = load_dataset()


def initialize_parameters_zeros(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    parameters = {}
    l = len(layer_dims)

    for i in range(1, l):
        parameters['W' + str(i)] = np.zeros((layer_dims[i], layer_dims[i - 1]))
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def initialize_parameters_random(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    parameters = {}
    l = len(layer_dims)
    np.random.seed(3)

    for i in range(1, l):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 10
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def initialize_parameters_he(layer_dims):
    """

    :param layer_dims:
    :return:
    """
    np.random.seed(3)
    parameters = {}
    l = len(layer_dims)

    for i in range(1, l):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(2./layer_dims[i -1])
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

    return parameters


def model(x, y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he'):
    """

    :param x:
    :param y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param initialization:
    :return:
    """
    grads = {}
    costs = []
    m = x.shape[1]
    layer_dims = [x.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layer_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layer_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layer_dims)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(x, parameters)
        cost = compute_loss(a3, y)
        grads = backward_propagation(x, y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i%100 == 0:
            print("cost after iteration {}:{}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.xlabel('iteration (per hundreds)')
    plt.ylabel('cost')
    plt.title('learning rate ='+str(learning_rate))
    plt.show()
    return parameters


if __name__ == '__main__':
    parameters = initialize_parameters_he([2, 4, 1])
    print ("W1 = " + str(parameters['W1']))
    print ("b1 = " + str(parameters['b1']))
    print ("W2 = " + str(parameters['W2']))
    print ("b2 = " + str(parameters['b2']))

    parameters = model(train_x, train_y, initialization="he")
    print("On the train set: ")
    predict_train = predict(train_x, train_y, parameters)
    print("On the test set: ")
    predict_test = predict(test_x, test_y, parameters)

    plt.title("Model with He initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x:predict_dec(parameters, x.T), train_x, train_y)