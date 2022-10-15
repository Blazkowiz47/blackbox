import numpy as np

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def relu(z):
    return np.maximum(z,0)

def drelu(z):
    return (z > 0) * 1

def dsigmoid(z):
    return z*(1-z)

def linear(z):
    return z

def dlinear(z):
    return 1

def get_activation(activation, diff = False):
    if activation == 'relu':
        return drelu if diff else relu 
    elif activation == 'sigmoid':
        return dsigmoid if diff else sigmoid
    else:
        return dlinear if diff else linear

