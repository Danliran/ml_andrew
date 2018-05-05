import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import ndimage
import scipy
from PIL import Image
from lr_utils import load_dataset
import pylab

def sigmod(z):
    s = 1/(1+np.exp(-z))
    return s;

def initialize_with_zero(dim):
    w = np.zeros((dim,1))
    b = 0

    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b, int))
    return w, b

def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmod(np.dot(w.T,X)+b) #compute activation
    cost = -(np.dot(Y,np.log(A.T))+np.dot(np.log(1-A),(1-Y).T))/m
    #BACKWARD PROPAGATION(TO FIND GRAND)
    dw =(np.dot(X,(A-Y).T))/m
    db =np.sum(A - Y)/m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost =np.squeeze(cost)
    assert (cost.shape ==())
    grads ={"dw": dw,
            "db": db
            }
    return grads, cost

def optimize(w ,b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        #update rule
        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i%100 == 0:
            costs.append(cost)
        if print_cost and i%100==0:
            print('Cost after iteration %i: %f' %(i, cost))

    params = {'w': w,
              'b':b}
    grads = {'dw':dw,
             'db':db}
    return params, grads, costs

def predict(w,b,X):
    m = X.shape[1] # test vect number
    Y_predict = np.zeros((1,m))
    w =  w.reshape(X.shape[0],1)

    A = sigmod(np.dot(w.T,X)+b)

    for i in range(A.shape[1]):
        if A[0][i] <= 0.5 : A[0][i]=0
        else: A[0][i]= 1
    Y_predict = A

    assert (Y_predict.shape == (1,m))

    return Y_predict

def model(X_train, Y_train, X_test, Y_test, num_iteration=1000, learning_rate = 0.5, print_cost =False):
    w,b = initialize_with_zero(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iteration, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print('train accuracy: {} %'.format(100 - np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Y_prediction_test-Y_test))*100))

    d = {'costs':costs,
         'Y_prediction_test':Y_prediction_test,
         'Y_prediction_train':Y_prediction_train,
         'w':w,
         'b':b,
         'learning_rate':learning_rate,
         'num_iteration':num_iteration}

    return d


if __name__ == '__main__':

    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
#    index=25
#    plt.imshow(train_set_x_orig[index])
#    print('y = ' + str(train_set_y[:,index]) + ', it`s a ' + classes[np.squeeze(train_set_y[:,index])].decode('utf-8') +' picture' )

    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px =  train_set_x_orig.shape[1]
#    print(train_set_x_orig.shape)
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    test_set_x_flatten= test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
#    print('train_set_x_flatten shape: '+ str(train_set_x_flatten.shape))
#    print('train_set_y shape: ' + str(train_set_y.shape))
#    print('test_set_x_flatten shape: ' + str(test_set_x_flatten.shape))
#    print('test_set_y shape: ' + str(test_set_y.shape))

    train_set_x = train_set_x_flatten /255
    test_set_x = test_set_x_flatten /255
#    print(train_set_x.shape)
    #w,b,X,Y = np.array([[1.],[2.]]), 2, np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

    #params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost = True)
   # print('w= ' + str(params['w']))
  #  print('b= ' + str(params['b']))
 #   print('dw= ' + str(grads['dw']))
#    print('db= ' + str(grads['db']))
    d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iteration=500, learning_rate=0.001,print_cost =True)
    #print('w= ' + str(d['w']))
    #print('b= ' + str(d['b']))
    '''
    learning_rate = [0.01,0.001,0.0001]
    module ={}
    for i in learning_rate:
        print("learning_rat is " + str(i))
        module[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iteration=1000, learning_rate=i,print_cost =False)
        print('\n'+'-----------------------------------------'+'\n')

    for i in learning_rate: 
        plt.plot(np.squeeze(module[str(i)]['costs']), label = str(module[str(i)]['learning_rate']))

    plt.ylabel('costs')
    plt.xlabel('iterations')
    legend = plt.legend(loc = 'upper center', shadow = True)
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    plt.show()
'''
    my_image = "test8.jpg"
    fname ="image/"+my_image
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
    my_predicated_image = predict(d['w'], d['b'],my_image)
    print(my_predicated_image)
    print(classes)
    plt.imshow(image)
    plt.show()
    print('y=' + str(np.squeeze(my_predicated_image))+", your algorithm predicts a  " + classes[int(np.squeeze(my_predicated_image)),].decode("utf-8")+" picture")








