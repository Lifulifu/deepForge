from keras.datasets import mnist, cifar10
import numpy as np


def load_mnist():
    (rawtrain, ytrain), (rawtest, ytest) = mnist.load_data()
    Ntrain = rawtrain.shape[0]
    Ntest = rawtest.shape[0]

    # zero-pad 28*28 img into 32*32 for convenience
    # and normalize
    rawtrain = np.pad(rawtrain, ((0,0),(2,2),(2,2)), 'constant') / 256.
    rawtest = np.pad(rawtest, ((0,0),(2,2),(2,2)), 'constant') / 256.

    #x = left half, y = right half
    xtrain = rawtrain[:, :, :16, np.newaxis]
    ytrain = rawtrain[:, :, 16:, np.newaxis]

    xtest = rawtest[:, :, :16, np.newaxis]
    ytest = rawtest[:, :, 16:, np.newaxis]

    return xtrain, xtest, ytrain, ytest