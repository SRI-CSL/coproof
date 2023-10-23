
import torch
import numpy as np
import PIL
import math
import random
import scipy

def ensure_torch(X, device="cpu"):
    if isinstance(X, torch.Tensor):
        return X.to(device)
    elif isinstance(X, np.ndarray):
        return torch.Tensor(X).to(device)
    elif isinstance(X, list):
        return [ensure_torch(device, x) for x in X]
    else:
        raise TypeError("Unknown type={}".format(type(X)))

def ensure_numpy(tensor):
    """ If the tensor is a torch one, detaches and converts to a numpy array.
    Otherwise returns as-is (implicitly a numpy structure).
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        # TODO: Ensure this is actually a valid numpy structure
        # If this is a sparse matrix, force it to be dense
        if scipy.sparse.issparse(tensor):
            return tensor.toarray()
        return tensor

def count_freqs(iterable):
    freqs = {}
    for item in iterable:
        freqs[item] = freqs.get(item, 0) + 1
    return sorted(list(freqs.items()), key=lambda x: x[1], reverse=True)
