import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    h = np.matmul(x,np.transpose(W1))
    h = np.maximum(0, h)

    y = np.maximum(0,np.matmul(h,np.transpose(W2)) + x)

    return y