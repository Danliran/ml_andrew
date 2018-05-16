# -*- coding: utf-8 -*-

from dnn_app_utils_v2 import *
from logic_regression.lr_utils import load_dataset


plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x:
    :param n_h:
    :param h_y:
    :return:
    """
    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2

    }
    return parameters


def initialize_parameters_deep(layer_dims):
    """
    :param layer_dims:
    :return:
    """
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1])*0.01
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))

        assert (parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layer_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    """

    :param A:
    :param W:
    :param b:
    :return:
    """
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)   # liner_cache=(A, W, b) activation_cache=Z  cache=(A, W, b, Z)

    return A, cache


def L_model_forward(X, parameters):
    """

    :param X:
    :param parameters:
    :return:
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for i in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(i)], parameters['b' + str(i)],
                                             activation="relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert (AL.shape == (1, X.shape[1]))
    return AL, caches


def compute_cost(AL, Y):
    """

    :param AL:
    :param Y:
    :return:
    """
    m = Y.shape[1]

    cost = -(np.dot(Y, np.log(AL.T)) + np.dot(1-Y, np.log(1 - AL).T))/m

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    return cost


def linear_backward(dZ, cache):
    """

    :param dZ:
    :param cache:
    :return:
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """

    :param dA:
    :param cache:
    :param activation:
    :return:
    """
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    """

    :param AL:
    :param Y:
    :param cache:
    :return:
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1-AL))

    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache,
                                                                                                  activation="sigmoid")
    for i in reversed(range(L-1)):
        current_cache = caches[i]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(i+2)], current_cache, "relu")
        grads["dA" + str(i+1)] = dA_prev_temp
        grads["dW" + str(i+1)] = dW_temp
        grads["db" + str(i+1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """

    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    L = len(parameters) // 2

    for i in range(L):
        parameters["W" + str(i+1)] = parameters["W" + str(i+1)] - learning_rate*parameters["W" + str(i+1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * parameters["b" + str(i + 1)]

    return parameters


def two_layer_model(X, Y, layers_dims, learning_rate=0.0001, num_iteration=3000, print_cost=False):
    """

    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iteration:
    :param print_cost:
    :return:
    """
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iteration):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")

        cost = compute_cost(A2, Y)

        dA2 = -(np.divide(Y, A2) - np.divide(1-Y, 1-A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1  = linear_activation_backward(dA1, cache1, "relu")

        grads["dW1"] = dW1
        grads["dW2"] = dW2
        grads["db1"] = db1
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i, np.squeeze(cost)))

        if i % 100 ==0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0001, num_iteration=3000, print_cost=False):
    """

    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iteration:
    :param print_cost:
    :return:
    """
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iteration):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}:{}".format(i, np.squeeze(cost)))

        if i % 100 ==0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iteration (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


if __name__ == "__main__":
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

    m_train = train_x_orig.shape[0]
    m_test = test_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    train_set_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_set_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    train_set_x = train_set_x_flatten / 255
    test_set_x = test_set_x_flatten / 255

    n_x = 12288
    n_h = 7
    n_y = 1
#    layers_dims = (n_x, n_h, n_y)
    layers_dims = [12288, 20, 7, 5, 1]

#    parameters = two_layer_model(train_set_x, train_y, layers_dims=layers_dims, num_iteration=2500, print_cost=True)

    L_parameters = L_layer_model(train_set_x, train_y, layers_dims, num_iteration=2500, print_cost=True)

