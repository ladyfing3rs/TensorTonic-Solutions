import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    W3 = np.array(W3, dtype=float)
    Ws = np.array(Ws, dtype=float)

    y1 = np.maximum(0, np.matmul(x,W1))
    y2 = np.maximum(0, np.matmul(y1, W2))
    y3 = np.matmul(y2, W3)

    if x.shape[1] == y3.shape[1]: 
        skip = x
    else:
        skip = np.matmul(x, Ws)    
    y = np.maximum(0, y3 + skip) 

    return y