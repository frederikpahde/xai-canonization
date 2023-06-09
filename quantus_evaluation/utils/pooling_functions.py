import numpy as np

def pool_pos_sq_sum(attr, **reduce_axes):
    return np.sum(attr.clip(min=0) ** 2, **reduce_axes)

def pool_sum(attr, **reduce_axes):
    return np.sum(attr, **reduce_axes)

def pool_max_norm(attr, **reduce_axes):
    return np.max(np.abs(attr), **reduce_axes)