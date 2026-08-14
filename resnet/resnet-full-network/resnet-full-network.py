import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.array(x, dtype = float)
    W1_b1 = np.array(W1_b1, dtype = float)
    W1_b2 = np.array(W1_b2, dtype = float)
    W2_b1 = np.array(W2_b1, dtype = float)
    W2_b2 = np.array(W2_b2, dtype = float)
    fc = np.array(fc, dtype=float)
    conv1 = np.array(conv1, dtype=float)
    
    x1 = x @ conv1
    x1 = np.maximum(0, x1)

    x2 = x1 @ W1_b1
    x2 = np.maximum(0, x2)
    x3 = x2 @ W2_b1
    x4 = x3 + x1
    x4 = np.maximum(0, x4)

    x5 = x4 @ W1_b2
    x5 = np.maximum(0, x5)
    x6 = x5 @ W2_b2
    x7 = x6 + x4 @ Ws_b2
    x7 = np.maximum(0, x7)

    y = x7 @ fc
    return y 
