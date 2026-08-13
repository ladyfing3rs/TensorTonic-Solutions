import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    Ws = np.array(Ws, dtype=float)

    h = np.matmul(x,W1)
    h = np.maximum(0, h)

    h = np.matmul(h,W2)

    y = h + np.matmul(x,Ws)

    y = np.maximum(0,y)

    return y