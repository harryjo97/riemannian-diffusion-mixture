import jax.numpy as np

from jax.numpy import index_exp as index
# from jax.ops import index, index_add, index_update


def to_ndarray(x, to_ndim, axis=0):
    x = np.array(x)
    if x.ndim == to_ndim - 1:
        x = np.expand_dims(x, axis=axis)

    if x.ndim != 0:
        if x.ndim < to_ndim:
            raise ValueError("The ndim was not adapted properly.")
    return x


def index_add(x, index, value):
    return x.at[index].add(value)


def index_update(x, index, value):
    return x.at[index].set(value)
