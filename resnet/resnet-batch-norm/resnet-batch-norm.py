import numpy as np

def post_activation_block(x, W1, W2, gamma1, gamma2, beta1, beta2):
    x1 = x @ W1
    mean = np.mean(x1, axis=0)
    var = np.var(x1, axis=0) + 1e-5
    var = np.sqrt(var)

    x1 = (x1 - mean) / var

    x2 = gamma1*x1 + beta1
    x2 = np.maximum(0, x2)

    x3 = x2 @ W2
    mean = np.mean(x3, axis=0)
    var = np.var(x3, axis=0) + 1e-5
    var = np.sqrt(var)

    x4 = (x3 - mean) / var
    x4 = gamma2*x4 + beta2

    y = np.maximum(0, x + x4)

    return y


def pre_activation_block(x, W1, W2, gamma1, gamma2, beta1, beta2):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0) + 1e-5
    var = np.sqrt(var)

    x1 = (x - mean) / var
    x1 = gamma1*x1 + beta1
    x1 = np.maximum(0, x1)

    x2 = x1 @ W1

    mean = np.mean(x2, axis=0)
    var = np.var(x2, axis=0) + 1e-5
    var = np.sqrt(var)

    x3 = (x2 - mean) / var
    x3 = gamma2*x3 + beta2
    x3 = np.maximum(0, x3)

    x4 = x3 @ W2

    y = x + x4

    return y
    
def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)

    if mode == "post":
        out = post_activation_block(x, W1, W2, gamma1, gamma2, beta1, beta2)
    else:
        out = pre_activation_block(x, W1, W2, gamma1, gamma2, beta1, beta2)
    return {"output": [[round(float(v), 4) for v in row] for row in out], "mode": mode}
