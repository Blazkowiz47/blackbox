import numpy as np
from utils.activation  import get_activation


def feed_forward(x, w, b, activation="linear"):
    '''
    x : Represents the input from the previous layer, Shape of x: (number_of_data_points, number_of_features),
    w : Weights of current layer, Shape of w: (number_of_features, output_features),
    b : Bais of current layer, Shape of b: (1, output_features).
    activation : tells the type of activation layer one requires with layer, default is linear.
    '''
    z = x@w + b
    a = get_activation(activation)(z)
    return z,a


def back_prop(a_prev, w, a, da_next, activation="linear"):
    '''
    a = output of current layer.
    '''
    dz = da_next * get_activation(activation,diff=True)(a)
    dw = a_prev.T @ dz
    db = dz.sum(axis = 0)
    da = dz @ w.T
    assert db.shape[0] == w.shape[1], f"Wrong db, expected: {w.shape[1]} got db = {db.shape[1]}"
    assert dw.shape == w.shape, f"Wrong dw, expected: {w.shape} got {dw.shape}"
    return dw.clip(-1000,1000), db.clip(-1000,1000), da.clip(-1000,1000)

