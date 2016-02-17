# data_utils.py
# @author: Lisa Wang
# @created: Feb 14 2016
#
#===============================================================================
# DESCRIPTION:
#
# A collection of functions that return train, val and test data for
# various datasets
#
#===============================================================================
# CURRENT STATUS: In progress
#===============================================================================
# USAGE: from data_utils import *
# 

import os, struct
import numpy as np

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros, int32

import cPickle as pickle
from scipy.misc import imread

def load_mnist():
  f = open('../data/mnist/mnist.pkl')
  train, test, val = pickle.load(f)
  f.close()
  X_train, y_train = change_to_array(train, 28, 28)
  X_val, y_val = change_to_array(val, 28, 28)
  X_test, y_test = change_to_array(test, 28, 28)
  return X_train, y_train, X_val, y_val, X_test, y_test

def change_to_array(M, H, W):
  N = len(M[0])
  X = np.array(M[0], dtype=float).reshape((N,1,H,W))
  y = np.array(M[1], dtype=int32)
  return X, y

def load_mnist_fail(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py

    :param string dataset: a string that denotes whether you want the training set or the testing set
    :param list digits (optional): use this if you only want data for a subset of digits.
    :param string path: the absolute/relative path of the directory where data is stored
    """
    if dataset == "training":
        fname_img = path + '/train-images-idx3-ubyte' #os.path.join(path, '/train-images-idx3-ubyte')
        fname_lbl = path + '/train-images-idx3-ubyte' #os.path.join(path, '/train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = path + '/t10k-images-idx3-ubyte' #os.path.join(path, '/t10k-images-idx3-ubyte')
        fname_lbl = path + '/t10k-labels-idx1-ubyte' #os.path.join(path, '/t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, 1, rows, cols), dtype=uint8)
    labels = zeros(N, dtype=int8)
    for i in range(len(ind)):
        images[i][0] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    for i in range(10):
      print i, len([x for x in range(size) if lbl[x] == i])
    print "WHEE", images.shape, labels.shape, N, digits, len(lbl)
    return images, labels


def load_cifar10(num_training=49000, num_validation=1000, num_test=1000):
    """
    WARNING: Needs to be run from code directory, otherwise relative path
    will not work.
    Load the CIFAR-10 dataset from disk.
    Returns train, validation and test sets.
    Note that num_training, num_validation and num_test have to be > 0.
    Adapted from CS231N assignment 1.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../data/cifar10'
    X_train, y_train, X_test, y_test = _load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    # X_train = X_train.reshape(num_training, -1)
    # X_val = X_val.reshape(num_validation, -1)
    # X_test = X_test.reshape(num_test, -1)

    X_train = X_train.transpose((0, 3, 1, 2))
    X_val = X_val.transpose((0, 3, 1, 2))
    X_test = X_test.transpose((0, 3, 1, 2))

    X_train = X_train.astype(np.int32)
    X_val = X_val.astype(np.int32)
    X_test = X_test.astype(np.int32)  

    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)            

    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_CIFAR10(ROOT):
  """ load all of cifar, adapted from CS231N assignment 1 """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = _load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = _load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def _load_CIFAR_batch(filename):
  """ load single batch of cifar, adapted from CS231N assignment 1"""
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


if __name__=='__main__':
	X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(num_training=49000, num_validation=1, num_test=1000)
	print 'X_train shape: {}'.format(X_train.shape)
	print 'X_val shape: {}'.format(X_val.shape)
	print 'X_test shape: {}'.format(X_test.shape)
