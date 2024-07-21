"""Jax based random backend. Inspired by https://github.com/wesselb/lab/blob/master/lab/jax/random.py """

from numpy.random import (  # NOQA
    # multivariate_normal,
#     normal,
    # rand,
    # randint,
    seed,
    # uniform,
    # choice
)

import jax
import sys

backend = sys.modules[__name__] 


def create_random_state(seed = 0):
    return jax.random.PRNGKey(seed=seed)


backend.jax_global_random_state = jax.random.PRNGKey(seed=0)


def global_random_state():
    return backend.jax_global_random_state


def set_global_random_state(state):
    backend.jax_global_random_state = state


def get_state(**kwargs):
    has_state = 'state' in kwargs
    state = kwargs.pop('state', backend.jax_global_random_state)
    return state, has_state, kwargs


def set_state_return(has_state, state, res):
    if has_state:
        return state, res 
    else:
        backend.jax_global_random_state = state
        return res


def _rand(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.uniform(key, size, *args, **kwargs)


def rand(size, *args, **kwargs):
    size = size if hasattr(size, "__iter__") else (size,)
    state, has_state, kwargs = get_state(**kwargs)
    state, res = _rand(state, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


uniform = rand


def _randint(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.uniform(key, size, *args, **kwargs)


def randint(size, *args, **kwargs):
    size = size if hasattr(size, "__iter__") else (size,)
    state, has_state, kwargs = get_state(**kwargs)
    state, res = _randint(state, size, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _normal(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.normal(key, size, *args, **kwargs)


def normal(size, *args, **kwargs):
    size = size if hasattr(size, "__iter__") else (size,)
    state, has_state, kwargs = get_state(**kwargs)
    state, res = _normal(state, size, *args, **kwargs)
    return set_state_return(has_state, state, res)
   

def _choice(state, a, n, *args, **kwargs):
    state, key = jax.random.split(state)
    inds = jax.random.choice(key, a.shape[0], (n,), replace=True, *args, **kwargs)
    choices = a[inds]
    return state, choices[0] if n == 1 else choices


def choice(a, n, *args, **kwargs):
    state, has_state, kwargs = get_state(**kwargs)
    state, res = _choice(state, a, n, *args, **kwargs)
    return set_state_return(has_state, state, res)


def _multivariate_normal(state, size, *args, **kwargs):
    state, key = jax.random.split(state)
    return state, jax.random.multivariate_normal(key, *args, **kwargs).sample(size)


def multivariate_normal(size, *args, **kwargs):
    size = size if hasattr(size, "__iter__") else (size,)
    state, has_state, kwargs = get_state(**kwargs)
    state, res = _multivariate_normal(state, size, *args, **kwargs)
    return set_state_return(has_state, state, res)
   