"""Jax based linear algebra backend."""

from jax._src import abstract_arrays
import jax.numpy as np
import numpy as _np
import functools
import scipy.linalg
from jax.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    norm,
    solve,
    svd,
    matrix_rank,
)
from jax import core, vmap, custom_vjp
from jax.scipy.linalg import expm as sp_expm

from .common import to_ndarray


def _is_symmetric(x, tol=1e-12):
    new_x = to_ndarray(x, to_ndim=3)
    return (np.abs(new_x - np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


def expm(x):
     x_new = to_ndarray(x, to_ndim=3)
     result = vmap(sp_expm)(x_new)
     return result[0] if len(result) == 1 else result


logm_prim = core.Primitive('logm')
logm_prim.def_impl(_np.vectorize(scipy.linalg.logm, signature='(m,n)->(m,n)'))
logm_prim.def_abstract_eval(lambda x: abstract_arrays.ShapedArray(x.shape, x.dtype))


def logm(x):
    return logm_prim.bind(x)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
        if np.all(a == b) and np.all(np.abs(a - np.transpose(a, axes)) < 1e-12):
            eigvals, eigvecs = eigh(a)
            if np.all(eigvals >= 1e-12):
                tilde_q = np.transpose(eigvecs, axes) @ q @ eigvecs
                tilde_x = tilde_q / (eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ np.transpose(eigvecs, axes)

    return np.vectorize(
        scipy.linalg.solve_sylvester, signature="(m,m),(n,n),(m,n)->(m,n)"
    )(a, b, q)


def sqrtm(x):
    return np.vectorize(scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)


def qr(*args, **kwargs):
    return np.vectorize(np.linalg.qr,
                         signature='(n,m)->(n,k),(k,m)')(*args, **kwargs)
