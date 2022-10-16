import numpy as np
from utils.activation  import get_activation


def feed_forward(x, w,b, activation=None):
    z = x@w + b
    a = get_activation(activation)(z)
    return z,a


def back_prop(a_prev, w, z, da_next, activation=None):
    dz = da_next * get_activation(activation,diff=True)(z)
    dw = a_prev.T @ dz
    db = dz.sum(axis = 0)
    da = dz @ w.T
    assert db.shape[0] == w.shape[1], f"Wrong db, expected: {w.shape[1]} got db = {db.shape[1]}"
    assert dw.shape == w.shape, f"Wrong dw, expected: {w.shape} got {dw.shape}"
    return dw.clip(-1000,1000), db.clip(-1000,1000), da.clip(-1000,1000)

