# Get CIFAR10
cd ../data
curl http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
mv cifar-10-batches-py cifar10
