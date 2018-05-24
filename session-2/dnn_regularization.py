# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *


plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x, train_y, test_x, test_y = load_2D_dataset()

def forward_propagation_with_dropout(x, parameters, keep_prob):
    return 0


def compute_cost_with_regularization(a3, y, parameters, lambd):
    """

    :param a3:
    :param y:
    :param parameters:
    :param lambd:
    :return:
    """
    m = y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(a3, y)
    l2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))/(2*m)

    cost = cross_entropy_cost + l2_regularization_cost

    return cost



def backward_propagation_with_regularization(x, y, cache, lambd):

    return 0


def backward_propagation_with_dropout(x, y, cache, keep_prob):
    return 0


def model(x, y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    """

    :param x:
    :param y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param lambd:
    :param keep_prob:
    :return:
    """
    grads = {}
    costs = []
    m = x.shape[1]
    layers_dims = [x.shape[0], 20, 3, 1]

    parameters = initialize_parameters(layers_dims)

    for i in range(0, num_iterations):
        if keep_prob == 1:
            a3, cache = forward_propagation(x, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(x, parameters, keep_prob)

        if lambd == 0:
            cost = compute_cost(a3, y)
        else:
            cost = compute_cost_with_regularization(a3, y, parameters, lambd)

        assert (lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(x, y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(x, y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(x, y, cache, keep_prob)

        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}:{}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    plt.plot(costs)
    plt.xlabel("iterations (x 1000)")
    plt.ylabel("cost")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()

    return parameters


if __name__ == '__main__':
    parameters = model(train_x, train_y)
    print("On the training set:")
    predictions_train = predict(train_x, train_y, parameters)
    print("On the test set:")
    predictions_test = predict(test_x, test_y,parameters)
    plt.title("Model without regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)
