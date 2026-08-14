import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    out = x 
    for grad_F in gradients_F:
        out = out + np.matmul(out, grad_F)
    return out

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    out = x 
    for grad_F in gradients_F:
        out = np.matmul(out, grad_F)
    return out
