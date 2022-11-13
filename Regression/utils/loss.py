import numpy as np

def mse(yt,yp):
    '''
    Mean Squared Error
    '''
    return ((yt-yp).square().mean())

def categorical_entropy(yt,yp):
    m = yt.shape[0]
    c1 = yt*np.log(yp)
    c2 = (1-yt) * np.log(1-yp)
    return ((c1.sum() + c2.sum()) / m, c1, c2)

def get_loss(loss):
    if loss == 'categorical_entropy':
        return categorical_entropy
    elif loss == 'msme':
        return mse
    else:
        print("Invalid loss function")
        raise NotImplementedError()
