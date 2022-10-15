import numpy as np
from Regression.activation  import get_activation

def feed_forward(x, w,b, activation):
    z = w@x.T + b
    a = get_activation(activation)(z)
    return z,a

def back_prop(a_prev, w, z, da_next, activation):
    dz = da_next*get_activation(activation,diff=True)(z)
    dw = dz*a_prev
    db = dz
    da = dz*w
    return dw, db, da

