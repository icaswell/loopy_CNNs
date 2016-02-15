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
from numpy import append, array, int8, uint8, zeros


def load_cifar10():
  	pass

def load_mnist(dataset="training", digits=np.arange(10), path="."):
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

    return images, labels

