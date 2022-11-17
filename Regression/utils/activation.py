import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    '''
    Parameters:
    z : a ndarray

    Returns :
    a ndarray of same shape as z, 
    Sigmoid is calculated element wise

    Sigmoid Function:
    https://en.wikipedia.org/wiki/Sigmoid_function
    '''
    a = 1/(1+np.exp(-z))
    return a

def relu(z):
    '''
    Relu Function:
    https://deepai.org/machine-learning-glossary-and-terms/relu
    '''
    return np.maximum(z,0)

def linear(z):
    '''
    NOTE: There is no activation function like linear, we have used it in the cases when there is no activation function is been used 
    in a layer, that's why we are just returning the same, without any changes.
    '''
    return z

def tanh(z):
    '''
    Tanh Function / HyperBolic Function
    https://paperswithcode.com/method/tanh-activation
    '''
    return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

def dsigmoid(a):
    '''
    Input : 
    a : output of sigmoid layer

    Differentiation of Sigmoid Function
    https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    '''
    return a*(1-a)

def drelu(a):
    '''
     Input : 
    a : output of sigmoid layer

    Differentiation of ReLU Function
    https://stats.stackexchange.com/questions/333394/what-is-the-derivative-of-the-relu-activation-function
    '''
    return (a > 0) * 1

def dlinear(a):
    '''
    Differntiation of Linear Function
    As been said that there is no acitvation like linear activation, the differentiation of this layer is one only.
    '''
    return np.ones(a.shape)

def dtanh(a):
    '''
    Differentiation of TanH Function
    https://blogs.cuit.columbia.edu/zp2130/derivative_of_tanh_function/
    '''
    return 1 - a**2


def get_activation(activation:str, diff = False):
    activation = activation.lower()
    if activation == 'relu':
        return drelu if diff else relu 
    elif activation == 'sigmoid':
        return dsigmoid if diff else sigmoid
    elif activation == 'tanh':
        return dtanh if diff else tanh
    elif activation == 'linear':
        return dlinear if diff else linear
    else: 
        raise NotImplementedError()


def visualization_helper(start:int, end: int, activation:str):
    z = np.linspace(start,end,num = 1000)
    acti = get_activation(activation=activation,diff = False)(z)
    dacti = get_activation(activation=activation,diff = True)(z)
    plt.plot(z,acti,'r')
    plt.plot(z,dacti,'b--')
    plt.title("Schematic Diagram of " + activation + " and derivative of " + activation)
    plt.legend([activation, "Derivative of " + activation])
    plt.grid()
    plt.show()

