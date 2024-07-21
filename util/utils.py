import jax
from jax import numpy as jnp
import numpy as np

def get_estimate_div_fn(fn):
    """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

    def div_fn(y, t, eps):
        eps = eps.reshape(eps.shape[0], -1)
        grad_fn = lambda y: jnp.sum(fn(y, t) * eps)
        grad_fn_eps = jax.grad(grad_fn)(y).reshape(y.shape[0], -1)
        return jnp.sum(grad_fn_eps * eps, axis=tuple(range(1, len(eps.shape))))

    return div_fn


def get_exact_div_fn(fn):
    "flatten all but the last axis and compute the true divergence"

    def div_fn(y, t):
        y_shape = y.shape
        dim = np.prod(y_shape[1:])
        t = jnp.expand_dims(t.reshape(-1), axis=-1)
        y = jnp.expand_dims(y, 1)  # NOTE: need leading batch dim after vmap
        t = jnp.expand_dims(t, 1)
        jac = jax.vmap(jax.jacrev(fn, argnums=0))(y, t)

        jac = jac.reshape([y_shape[0], dim, dim])
        return jnp.trace(jac, axis1=-1, axis2=-2)

    return div_fn


def div_noise(rng, shape, hutchinson_type):
    """Sample noise for the hutchinson estimator."""
    if hutchinson_type == "Gaussian":
        epsilon = jax.random.normal(rng, shape)
    elif hutchinson_type == "Rademacher":
        epsilon = (
            jax.random.randint(rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2
            - 1
        )
    elif hutchinson_type == "None":
        epsilon = None
    else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")
    return epsilon


def get_div_fn(drift_fn, hutchinson_type="None"):
    """Euclidean divergence of the drift function."""
    if hutchinson_type == "None":
        return lambda y, t, eps: get_exact_div_fn(drift_fn)(y, t)
    else:
        return lambda y, t, eps: get_estimate_div_fn(drift_fn)( y, t, eps)


def get_riemannian_div_fn(func, hutchinson_type="None", manifold=None):
    """divergence of the drift function.
    if M is submersion with euclidean ambient metric: div = div_E
    else (in a char) div f = 1/sqrt(g) \sum_i \partial_i(sqrt(g) f_i)
    """
    sqrt_g = (
        lambda x: 1.0
        if manifold is None or not hasattr(manifold.metric, "lambda_x")
        else manifold.metric.lambda_x(x)
    )
    drift_fn = lambda y, t: sqrt_g(y) * func(y, t)
    div_fn = get_div_fn(drift_fn, hutchinson_type)
    return lambda y, t, eps: div_fn(y, t, eps) / sqrt_g(y)


def get_pode_drift(mix, modelf_w_dicts, modelb_w_dicts):
    modelf, paramsf, statesf = modelf_w_dicts
    modelb, paramsb, statesb = modelb_w_dicts
    def drift_fn(y, t):
        """The drift function of the reverse-time SDE."""
        #NOTE: add projection
        y = mix.manifold.projection(y)
        driftf = mix.get_drift_fn(modelf, paramsf, statesf)
        driftb = mix.rev().get_drift_fn(modelb, paramsb, statesb)
        pode = mix.probability_ode(driftf, driftb)
        return pode.coefficients(y, t)[0]

    return drift_fn
    