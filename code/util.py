#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# File: generic_util.py
# @author: Isaac Caswell
# @created: 21 February 2015
# File: util.py
# @author: Isaac Caswell, Lisa Wang
#===============================================================================
# DESCRIPTION:
#
# Generically useful/organizational functions
#
#===============================================================================
# CURRENT STATUS: Works!  In progress.
#===============================================================================
# USAGE:
# import util
# util.colorprint("This text is flashing in some terminals!!", "flashing")
# 

#standard modules
import numpy as np
import time
from collections import Counter, defaultdict
import heapq
import matplotlib.pyplot as plt
import argparse
import shutil
import csv
import os
import re
import collections
import json
import hashlib
import struct

from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

#===============================================================================
# FUNCTIONS
#===============================================================================


#-----------------------------------------------------------------------------------------            


def plot_loss_acc(losses, train_accs, val_accs, xlabel, attributes):   
    """
    This function plots the loss, the train accuracies and the validation
    accuracies. losses, train_accs and val_accs are expected to be lists of 
    the same length.
    Input:

    -   losses:list of losses
    -   train_accs: list of train accuracies
    -   val_accs: list of validation accuracies
    -   xlabel: string describing what the label of the x axis should be
    -   attributes: a dictionary containing the attributes describing the
        data set, hyperparameters and other factors pertaining to this
        particular run. 
        E.g.
            -   lr: learning rate 
            -   rg: regularization strength 
            -   ep: number of epochs
            -   data_set_name: a string with the name of the dataset
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    train_accs_line, = ax1.plot(xrange(len(train_accs)), train_accs, 'b-', label='train accuracies')
    val_accs_line, = ax1.plot(xrange(len(val_accs)), val_accs, 'g-', label='val accuracies')

    ax1.set_ylabel('accuracies', color='b')
    ax1.set_xlabel(xlabel)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    losses_line, = ax2.plot(xrange(len(losses)), losses, 'r-', label='losses')
    ax2.set_ylabel('losses', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

    plt.legend(handles=[losses_line, train_accs_line, val_accs_line],bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand")
    
    figure_filename = '../results/loss_plots/' + descriptive_filename(attributes, extension = '.png', random_stamp=True)
    fig.savefig(figure_filename)

def colorprint(message, color="rand", newline=True):
    """
    '\033[%sm' + message +'\033[0m',
    """
    message = str(message)
    if color == "bare":
        print message;
        return
    # message = unicode(message)
    """prints your message in pretty colors! So far, only a few color are available."""
    if color == 'none': print message,
    if color == 'demo':
        for i in range(-8, 109):
            print '\n%i-'%i + '\033[%sm'%i + message + '\033[0m\t',
        return
    print '\033[%sm'%{
        'bold' : 1,
        'grey' : 2,
        'underline' : 4,
        'flashing' : 5,
        'black_highlight': 7,
        'invisible' : 8,
        'black' : 30,
        'purple' : 34,      
        'peach' : 37,
        'red_highlight' : 41,
        'green_highlight' : 42,
        'orange_highlight' : 43,
        'blue_highlight' : 44,
        'magenta_highlight' : 45,
        'teal_highlight' : 46,
        'lavendar_highlight' : 47,        
        'graphite' : 90,
        'red' : 91,
        'green' : 92,
        'yellow' : 93,  
        'blue' : 94,
        'magenta' : 95,
        'teal' : 96,
        'pink' : 97,
        'neutral' : 99,
        'rand' : np.random.choice(range(90, 98) + range(40, 48)+ range(30, 38)+ range(0, 9)),
    }.get(color, 1)  + message + '\033[0m',
    if newline:
        print '\n',


def print_matrix(mat, delimitor="\t", precision=3, color="bare", newline=False):
    """
    prints the elements in a matrix in a nice way!
    """
    assert len(mat.shape) <= 2
    if len(mat.shape) == 1:
        mat = mat.reshape((1, mat.shape[0]))
    res = ""
    val_format = "%." + str(precision) + "f"
    for row in mat:
        res += delimitor.join(val_format%val for val in row)
        res += "\n"
    colorprint(res, color, newline=newline)

def time_string(precision='day'):
    """ returns a string representing the date in the form '12-Jul-2013' etc.
    intended use: handy naming of files.
    """
    t = time.asctime()
    precision_bound = 10 #precision == 'day'
    yrbd = 19
    if precision == 'minute':
        precision_bound = 16
    elif precision == 'second':
        precision_bound = 19
    elif precision == 'year':
        precision_bound = 0
        yrbd = 20
    t = t[4:precision_bound] + t[yrbd:24]
    t = t.replace(' ', '-')
    return t


def random_string_signature(length = 4):
    candidates = list("qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890")
    np.random.shuffle(candidates)
    return "".join(candidates[0:length])


def descriptive_filename(attributes, 
                         omit={},
                         id="", 
                         truncate_decimals=0, 
                         shorten=False, 
                         timestamp_precision='day', 
                         random_stamp=False, 
                         extension = '.out'
                         ):
    """
    produces a nice, descriptive filename for a dict-like that specifies the attributes of a run.
    Also offers the ability to add a random descriptor to prevent filename collisions, and the 
    option to shorten it to something less human readable but less annoying.

    EXAMPLES:
    -------------------------------------------------------------------------------
    # default
    alpha=0.5_optimizer=rmsprop_nu=1.6e-11_epsilon=0.001__Dec-10-2015.out 

    # adding id "toy-test" and truncating decimals to three decimal places
    toy-test__alpha=0.500_optimizer=rmsprop_nu=0.000_epsilon=0.001__Dec-10-2015.out

    # adding id "toy-test" and omitting the parameter "epsilon"
    toy-test__alpha=0.5_optimizer=rmsprop_epsilon=0.001__Dec-10-2015.out

    # adding id "toy-test" and using the option shorten=True
    94dd2__Dec-10-2015.out

    # iusing a random identifier and a custom extension ".lol"
    alpha=0.5_optimizer=rmsprop_nu=1.6e-11_epsilon=0.001__xIFm__Dec-10-10:07:04-2015.lol
    -------------------------------------------------------------------------------    

    :param dict attributes: a dict OR an argparse.Namespace that maps parameter names to their values 
    :param iterable omit: an iterable specifying parameters to exclude from the filename
    :param str id: an id (e.g. to identify a particular set of tests) that will be prepended to the result
    :param int truncate_decimals: if this is nonzero, all floats mapped to by attributes will be truncated 
            at this decimal count.  The problem with this is that numbers like 1e-6 will turn into 
            e.g. 0.000, and 5.0 will turn into 5.000
    :param bool shorten: if True, the filename will be a hash of the string describing the attributes, 
            rather than the string itself.  The reason for this is that if you have a lot of attributes,   
            the filename could get maddeningly long.  timestamp is still added after this.
    :param str timestamp_precision: the precision of the timestamp.  May be "day", "minute" or "second".  
            If it is False or "", no timestamp is appended to the filename.
    :param bool random_stamp: whether to add a random stamp at the end of the filename.  This should be 
            used if multiple runs with the same parameters want different filenames (e.g. so the results
            of several runs are averaged)
    :param str extension: what extension to use for the file.  Must begin in '.' or be the empty string 
            (signifying no extension)

    """
    if isinstance(attributes, argparse.Namespace):
        attributes = dict(attributes._get_kwargs());

    res = "" if not id else id+"__"
    #--------------------------------------------------
    # get the identifier from the attribute dict
    # 
    complete_identifier = ""
    for key, val in attributes.items():
        if key in omit: continue
        strval = str(val)
        if truncate_decimals and isinstance(val, float):
            strval = ("%.{0}f".format(truncate_decimals))%(val)
        complete_identifier += "%s=%s_"%(key, strval)
    complete_identifier = complete_identifier[0:-1] #remove trailing underscore
    if shorten:
        hashed_id = hashlib.sha1(complete_identifier).hexdigest()
        complete_identifier = str(hashed_id)[0:5]
    res += complete_identifier
    #--------------------------------------------------
    # add random stamp and timestamp, if wanted
    if random_stamp:
        res += '__' + random_string_signature(length=4)
    if timestamp_precision:
        res += '__' + time_string(timestamp_precision)

    if extension:
        assert extension[0] == '.'
        res += extension


    return res

    


def make_toy_data(outfile, dim=5, n_examples=100, misclassification_prob=0.1, predictor="dot", seeds=None):
    """
    creates a file of artificial data, where each line is an example (space separated values) 
    followed by a tab and then the true label

    :param string outfile: the file to which to write the data
    :param int dim: the dimensionality of the training examples that are to be made
    :param int n_examples: the number of examples to be made
    :param float misclassification_prob: this number of examples will be intentionally misclassified.
            right now this does not actually work--I just take the negative of the vector (works for dot but little else)
    :param function or str hypothesis: either a predictor that takes as input a numpy vector and 
    :param tuple seeds: a tuple of 0. the seed used to create the weight vectors, and 1. the seed for the examples.
            The reason for this is so that we can make different datasets (e.g. train and test) with the same weights
    returns an int or string, or a string representing one of our pre-built predictors.
    """
    if seeds: 
        np.random.seed(seeds[0])
    if isinstance(predictor, str):
        w = 2*(np.random.random((dim,)) - 0.5) # used for dot and pathological
        w2 = 2*(np.random.random((dim,)) - 0.5) # used only in pathological
        if predictor=="dot":
            predictor = lambda x: 1 if x.dot(w) > 0 else 0
        elif predictor=="sigmoid":
            predictor = lambda x: 1 if 1.0/(1.0 + np.exp(x.dot(w))) > 0.5 else 0
        elif predictor=="pathological":
            def p(x):
                a = np.log(np.outer(x, x + w2)**2).dot(x)
                a += x.dot(x**2 + w)
                # a **= 1.3
                a = a.dot(w)
                return 1 if a > -2 else 0 
            predictor = p
        else:
            print "please specify 'sigmoid' or 'dot'"
            return
    if seeds: 
        np.random.seed(seeds[1])            
    with open(outfile, "w") as f:
        labels = []
        for _ in range(n_examples):
             x = 2*(np.random.random((dim,)) - 0.5)
             y = predictor(x)
             labels.append(y)
             if np.random.rand() < misclassification_prob:
                x *=-1
             line_to_write = " ".join(str(val) for val in x) + "\t" + str(y) + "\n"
             f.write(line_to_write)
    print "class balance: ", Counter(labels)           

#===============================================================================
# TESTING SCRIPT
#===============================================================================

#-------------------------------------------------------------------------------
# part (i) 


if __name__ == '__main__':
    print 'You are running this script (util.py) as main, so you must be demoing it!'

    print "NOW FOR A DEMO OF HOM YOU CAN MAKE NICE, HYPERDESCRIPRIVE FILENAMES FOR YOUR MILLION RUNA OF THAT SCRIPT"
    run_attributes = {"alpha": 0.5, "epsilon": 0.001, "optimizer": "rmsprop", "nu": 16e-12}
    omit = {"nu"}
    print descriptive_filename(run_attributes)
    print descriptive_filename(run_attributes, id="toy-test", truncate_decimals=3)
    print descriptive_filename(run_attributes, id="toy-test", omit=omit)    
    print descriptive_filename(run_attributes, shorten=True)
    print descriptive_filename(run_attributes, timestamp_precision="second") 
    print descriptive_filename(run_attributes, timestamp_precision="second", random_stamp=True, extension=".lol")

    make_toy_data(outfile="../data/toy_data_5d.txt", dim=5, n_examples=100, misclassification_prob=0.0, predictor="dot", seeds=(2,3))

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-stuff', type=str, default="yeah")
    # ARGS = parser.parse_args()
    # print descriptive_filename(ARGS)     



    colorprint("nyaan", color="demo")           








