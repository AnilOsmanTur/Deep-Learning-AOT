# Get CIFAR10
#wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#tar -xzvf cifar-10-python.tar.gz
#rm cifar-10-python.tar.gz 

# Get MNIST
mkdir mnist
cd mnist/
rm *ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
