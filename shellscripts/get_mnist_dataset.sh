#!/bin/bash
cd ../data
mkdir mnist
cd mnist
wget http://deeplearning.net/data/mnist/mnist.pkl.gz
gzip -d mnist.pkl.gz
