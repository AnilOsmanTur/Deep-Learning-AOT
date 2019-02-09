#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:09:51 2018

@author: anilosmantur
"""

import random
import numpy as np
from metu.data_utils import load_dataset
import matplotlib.pyplot as plt
from cs231n.classifier_trainer import ClassifierTrainer
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.classifiers.convnet import *


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

import tables
from scipy import signal

filename='./metu/dataset/Part_1.mat'
window=1000
stride=1
width_limit=50
ppg_ind = 0
abg_ind = 1

X = []
Y = []

file_pt = tables.open_file(filename)
data_heap = file_pt.root.Part_1
print(data_heap.shape)
train_size = data_heap.shape[0]

sampleSizes = []
for ind in range(train_size):
    data = data_heap[ind][0][0]
    #print(data[:,abg_ind].shape[0])
    sample_size = data[:,abg_ind].shape[0]
    sampleSizes.append(sample_size)
    del data

sampleSizes = np.array(sampleSizes)
print(np.min(sampleSizes), np.max(sampleSizes))

data = data_heap[20][0][0][:,abg_ind][:1000]
padded = np.pad(data, [(12,12)], 'constant')
data2 = np.reshape(padded, (32,32))
print(np.sqrt(256))
print(1000/125)
plt.imshow(data2.T)
plt.show()

plt.plot(data)
plt.show()


    #print(sample_size-window)
    for t in range(0, sample_size-window, stride):
        print("Progress {}/{} in {}/{}".format(ind, train_size, t,sample_size-window), end="\r")
        #print(data[:,ppg_ind][t:t+window].shape)        
        X.append(data[:,ppg_ind][t:t+window])
        widths = np.arange(1, width_limit)
        inds1 = signal.find_peaks_cwt(data[:,abg_ind][t:t+window], widths)
        inds2 = signal.find_peaks_cwt(-data[:,abg_ind][t:t+window], widths)
        
        mean1 = np.mean(data[:,abg_ind][inds1])
        mean2 = np.mean(data[:,abg_ind][inds2])
        Y.append([mean1, mean2])
X = np.array(X)
Y = np.array(Y)


file_pt.close()
return X, Y

datafile = './metu/dataset/Part_1.mat'
X, y = load_dataset(datafile)

# Load the PPG dataset
# If your memory turns out to be sufficient, try loading a subset
def get_data(datafile,
             training_ratio=0.9,
             test_ratio=0.06,
             val_ratio=0.01,
             window=input_size,
             width_limit=50,
             stride=750):
    # Load the PPG training data stride
    X, y = load_dataset(datafile, window=window, stride=stride, width_limit=width_limit)

    # TODO: Split the data into training, validation and test sets
    length=len(y)
    num_training=int(length*training_ratio)
    num_val = int(length*val_ratio)
    num_test = min((length-num_training-num_val), int(length*test_ratio))
    mask = range(num_training-1)
    X_train = X[mask]
    y_train = y[mask]
    mask = range(num_training, num_training+num_test)
    X_test = X[mask]
    y_test = y[mask]
    mask = range(num_training+num_test, num_training+num_test+num_val)
    X_val = X[mask]
    y_val = y[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


datafile = './metu/dataset/Part_1.mat' #TODO: PATH to your data file
input_size = 1000 # TODO: Size of the input of the network

X_train, y_train, X_val, y_val, X_test, y_test = get_data(datafile, window=input_size, width_limit=50, stride=750)
print ("Number of instances in the training set: ", len(X_train))
print ("Number of instances in the validation set: ", len(X_val))
print ("Number of instances in the testing set: ", len(X_test))


net = PPGconvnet()
model = net.model

out = net.ppg_convnet(X[:10], model, y[:10])

del model