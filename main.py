#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
import time
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset

"""Computes the softmax matrix (c * n) corresponding to the 2D matrix X (c * n)"""
def softmax(X):
    c, n = X.shape
    # scale matrix down to avoid overflow
    Z = X - np.amax(X, axis=0)
    mat = np.zeros((c,n))
    sum = np.zeros(n)
    for k in range(c):
        sum = sum + np.exp(Z[k])
    for i in range(c):
        mat[i] = np.true_divide(np.exp(Z[i]), sum)
    return mat

# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    c, d = W.shape
    new_Xs = np.empty((d,len(ii)))
    new_Ys = np.empty((c,len(ii)))
    # create subset of the examples and labels
    # with indexes in vector ii
    for idx, j in enumerate(ii):
        new_Xs[:,idx] = Xs[:,int(j)]
        new_Ys[:,idx] = Ys[:,int(j)]
    grad = np.zeros((c,d))
    product = np.matmul(W,new_Xs)
    grad = np.matmul((softmax(product)-new_Ys),new_Xs.T)
    return (np.true_divide(grad,len(ii)) + gamma*W)

# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    start_logreg_error = time.time()
    c,d = W.shape
    _,n = Xs.shape
    Ys = Ys.astype(int)
    sftmx_preds = softmax(np.matmul(W,Xs))
    preds_arg  = np.argmax(sftmx_preds,axis=0)
    preds = np.zeros([c,n])
    count = 0
    for ii in range(n):
        preds[preds_arg[ii],ii] = int(1.0)
        if np.array_equal(preds[:,ii], Ys[:,ii]):
            count += 1
    error = (n-count)/n;
    assert(error>=0.)
    end_logreg_error = time.time()
    # print("Logreg Error Time: " + str(end_logreg_error - start_logreg_error))
    return error

# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    sftmx_preds = softmax(np.matmul(W,Xs))
    loss = (np.sum(-Ys*np.log(sftmx_preds)) + gamma*W)
    return loss

# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    grad = lambda Xs, Ys, gamma, W: multinomial_logreg_grad_i(Xs, Ys, range(n), gamma, W)
    _,n = Xs.shape
    W_update = []
    W = W0
    tt = 0
    while tt < num_epochs:
        W = W-alpha*grad(Xs, Ys, gamma, W)
        if tt%monitor_period == 0:
            W_update.append(W)
            tt+=1
        else:
            tt += 1
    return np.array(W_update)

# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    _,n = Xs.shape
    grad = lambda Xs, Ys, gamma, W: multinomial_logreg_grad_i(Xs, Ys, range(n), gamma, W)
    W_update = []
    W = W0
    V_old = W0
    iteration = 0
    for t in range(num_epochs):
        V_new = W - alpha*grad(Xs, Ys, gamma, W)
        W = V_new + beta*(V_new-V_old)
        V_old = V_new
        if t%monitor_period == 0:
            W_update.append(W)
        t += 1
    return np.array(W_update)

# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    _,n = Xs.shape
    grad = lambda Xs, Ys, ii, gamma, W: multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W)
    W_update = []
    W = W0
    iteration = 0
    for t in range(num_epochs):
        ii = []
        start_index = np.random.randint(0, high=n)
        for i in range(int(n/B)-1):
            index = (start_index + i) % n
            ii.append(index)
        if iteration%monitor_period == 0:
            W_update.append(W)
        iteration+=1
        W = W-alpha*grad(Xs, Ys, np.array(ii), gamma, W)
    return np.array(W_update)

# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    pass

# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    pass

if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()

    # shrink data set for faster debugging
    Xs_tr = Xs_tr[:,:1000]
    Ys_tr = Ys_tr[:,:1000]
    Xs_te = Xs_te[:,:1000]
    Ys_te = Ys_te[:,:1000]

    # _________ PART 1 ___________

    gamma = 0.0001
    alpha = 1.0
    num_epochs = 100
    monitor_period = 1
    beta1 = 0.9
    beta2 = 0.99

    W0 = np.random.rand(len(Ys_tr[:,0]),len(Xs_tr[:,0]))

    GD_W_iter = gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)
    Nesterov1_W_iter = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta1, num_epochs, monitor_period)
    Nesterov2_W_iter = gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta2, num_epochs, monitor_period)

    GD_train_err = []
    Nesterov1_train_err = []
    Nesterov2_train_err = []

    for i in range(GD_W_iter.shape[0]):
        GD_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, GD_W_iter[i]))
        Nesterov1_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, Nesterov1_W_iter[i]))
        Nesterov2_train_err.append(multinomial_logreg_error(Xs_tr, Ys_tr, Nesterov2_W_iter[i]))

    GD_train_err = np.array(GD_train_err)
    Nesterov1_train_err = np.array(Nesterov1_train_err)
    Nesterov2_train_err = np.array(Nesterov2_train_err)
