#-*- coding: UTF-8 -*-

import numpy as sp
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_extra_datasets, load_planar_dataset

np.random.seed(1)

def layer_sizes(X, Y):
    n_x =  X.shape[0]
    n_h =4
    n_y = Y.shape[0]

    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {
        'w1': w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
    }
    return parameters

def forward_propagation(X, parameters):
    w1 = parameters['w1']  #shape(nh,nx)
    b1 = parameters['b1']  #shape nh,1
    w2 = parameters['w2']  #shape ny,nh
    b2 = parameters['b2']  #shape ny, 1

    z1 = np.dot(w1,X) + b1
    a1 = np.tan(z1)
    z2 = np.dot(w2,a1) +b2
    a2 = sigmoid(z2)

    cache ={
        'z1':z1,
        'a1':a1,
        'z2':z2,
        'a2':a2
    }
    return a2, cache

def compute_cost(a2, Y, parameters):
    m = Y.shape[1]
    logprobes = np.multiply(np.log(a2), Y) +np.multiply(np.log(1-a2), 1-Y)
    cost = -np.sum(logprobes)/m
    cost = np.squeeze(cost)
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']

    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 -Y
    dw2 = np.dot(dz2, a1.T)/m
    db2 = np.sum(dz2, axis=1,keepdims=True)/m
    dz1 = np.dot(w2.T, dz2)*(1-np.power(a1,2))
    dw1 = np.dot(dz1, X.T)/m
    db1 = np.sum(dz1,axis=1,keepdims=True)/m

    grads={
        'dw1':dw1,
        'db1':db1,
        'dw2':dw2,
        'db2':db2
    }
    return  grads

def update_parameters(parameters, grads, learning_rate=1.2):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters={
        'w1':w1,
        'b1':b1,
        'w2':w2,
        'b2':b2
    }
    return parameters

def nn_model(X,Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    for i in range(0, num_iterations):
        a2, cache = forward_propagation(X, parameters)
        cost = compute_cost(a2, Y, parameters)

        grads = backward_propagation(parameters,cache, X, Y)

        parameters= update_parameters(parameters, grads)
        if print_cost and i%100 ==0:
            print("cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):
    a2, cache = forward_propagation(X, parameters)
    predictions = np.round(a2)
    return predictions





if __name__ == '__main__':
    print("planar test!")
    X, Y = load_planar_dataset()
    parameters = nn_model(X, Y, n_h=4,num_iterations = 1, print_cost=True)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidder layer size" + str(4))
    plt.show()

    predictions = predict(parameters, X)
    print('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)+'%')

'''
#    plt.scatter(X[0,:], X[1,:], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
 #   plt.show()
    shape_X = X.shape
    shape_Y = Y.shape

    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,Y.T)
    plot_decision_boundary(lambda x: clf.predict(x), X,Y)
    plt.title("Logitstic Regression")
    plt.show()

    LR_prediction = clf.predict(X.T)
    print('Accurary of logistic regression: %d' % float((np.dot(Y, LR_prediction) + np.dot(1-Y, 1-LR_prediction))/float(Y.size)*100)+ '%' + ' percentage of correctly labelled datapoint')
'''



