# import scipy.io as sio # you may use scipy for loading your data ** useless
# import cPickle as pickle # or Pickle library ** useless
import tables
import numpy as np
from scipy import signal
import os


def load_dataset(filename, window=1000, stride=1, width_limit=50):
    """Load your 'PPG to blood pressure' dataset"""
	# TODO: Fill this function so that your version of the data is loaded from a file into vectors
    ppg_ind = 0
    abg_ind = 1

    X = []
    Y = []

    file_pt = tables.open_file(filename)
    data_heap = file_pt.root.Part_1
    train_size = 500 #data_heap.shape[0]   
            
    for ind in range(train_size):
        data = data_heap[ind][0][0]
        #print(data[:,abg_ind].shape[0])
        sample_size = data[:,abg_ind].shape[0]
        #print(sample_size-window)
        print('\r', ' '*30, end='')
        sample_n = int((sample_size-window)/stride) + 1
        for i in range(sample_n):
            print("\rProgress {}/{} in {}/{}".format(ind + 1, train_size, i+1,sample_n), end="")
            #print(data[:,ppg_ind][t:t+window].shape)
            t = i * stride
            X.append(data[:,ppg_ind][t:t+window])
            widths = np.arange(1, width_limit)
            inds1 = signal.find_peaks_cwt(data[:,abg_ind][t:t+window], widths)
            inds2 = signal.find_peaks_cwt(-data[:,abg_ind][t:t+window], widths)
            
            mean1 = np.mean(data[:,abg_ind][inds1])
            mean2 = np.mean(data[:,abg_ind][inds2])
            Y.append([mean1, mean2])
    X = np.array(X)
    Y = np.array(Y)
    print('data loading done.')

    file_pt.close()
    return X, Y



if __name__ == '__main__':
	# TODO: You can fill in the following part to test your function(s)/dataset from the command line
	filename='./dataset/Part_1.mat'
	X, Y = load_dataset(filename)

