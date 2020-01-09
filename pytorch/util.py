from keras.datasets import mnist
import numpy as np
from torch.autograd import Variable
import torch
import random

def load_mnist():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    # pad to 32*32 and normalize to 0~1
    xtrain = np.pad(xtrain, ((0,0),(2,2),(2,2)), 'constant') / 255
    xtest = np.pad(xtest, ((0,0),(2,2),(2,2)), 'constant') / 255
    # expand channel dim
    xtrain, xtest = xtrain[:, np.newaxis, :, :, ], xtest[:, np.newaxis, :, :]

    return xtrain, ytrain, xtest, ytest

def onehot(x, size, exclusive=False):
    result = np.zeros((x.size(0), size))
    if not exclusive:
        num_list = [ int(num.item()) for num in x ]
        result[np.arange(x.size(0)), num_list] = 1
    else: 
        exclusive_list = [random.choice( list(range(1, int(num.item()))) + list(range(int(num.item())+1,size)) ) for num in x ]
        result[np.arange(x.size(0)), exclusive_list] = 1
    result = torch.tensor(result)
    return result

def reverse_onehot(x):
    indice = x.nonzero()[:,1].view(-1)
    return indice