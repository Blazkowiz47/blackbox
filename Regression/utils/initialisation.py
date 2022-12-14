import numpy as np


def DenseLayer(input_dim, output=1, initialiser='random', bias_initialiser='zero'):
    return __initialiser__(initialiser, (input_dim,output)), __initialiser__(bias_initialiser, (1,output))



def __initialiser__(initialiser, shape):
    if initialiser == 'zero':
        return np.zeros(shape)
    else:
        return np.random.rand(*shape)  