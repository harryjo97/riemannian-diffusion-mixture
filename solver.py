import abc
import jax
from jax import numpy as jnp
from util.registry import register_category

get_predictor, register_predictor = register_category("predictors")
get_corrector, register_corrector = register_category("correctors")


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde):
        super().__init__()
        self.sde = sde
        self.manifold = sde.manifold

    @abc.abstractmethod
    def update_fn(self, x, t, dt):
        """One update of the predictor.
        """
        raise NotImplementedError()


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x0, x, t):
        """One update of the corrector.
        """
        raise NotImplementedError()


@register_predictor
#NOTE: Geodesic Random Walk
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde):
        super().__init__(sde)

    def update_fn(self, rng, x, t, dt):
        z = self.sde.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )[1].reshape(x.shape[0], -1)
        drift, diffusion = self.sde.coefficients(x, t)

        tangent_vector = jnp.einsum(
            "...,...i,...->...i", diffusion, z, jnp.sqrt(jnp.abs(dt))
        )
        
        tangent_vector = tangent_vector + drift * dt[..., None]
        x = self.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


@register_corrector
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, snr, n_steps):
        pass
    def update_fn(self, rng, x0, x, t):
        return x, x


def get_pc_sampler(
        sde, 
        N, 
        predictor="EulerMaruyamaPredictor", 
        corrector="NoneCorrector", 
        snr=0.0, n_steps=1, eps=1.0e-3
    ): 
    """Create a Predictor-Corrector (PC) sampler.
    """
    assert sde.approx
    predictor = get_predictor(predictor)(sde)
    corrector = get_corrector(corrector)(
        sde, snr, n_steps
    )

    def pc_sampler(rng, x):
        t0 = jnp.broadcast_to(sde.t0, x.shape[0])
        tf = jnp.broadcast_to(sde.tf, x.shape[0])
        timesteps = jnp.linspace(start=t0, stop=tf-eps, num=N, endpoint=True)
        dt = (tf - t0) / N

        def loop_body(i, val):
            rng, x, x_mean, x_hist = val
            t = timesteps[i]
            rng, step_rng = jax.random.split(rng)
            x, x_mean = corrector.update_fn(step_rng, x0, x, t)
            rng, step_rng = jax.random.split(rng)
            x, x_mean = predictor.update_fn(step_rng, x, t, dt)
            x_hist = x_hist.at[i].set(x)
            return rng, x, x_mean, x_hist

        x_hist = jnp.zeros((N, *x.shape))
        x0 = x
        _, x, x_mean, x_hist = jax.lax.fori_loop(0, N, loop_body, (rng, x, x, x_hist))
        return x_mean 

    return pc_sampler


class EulerMaruyamaTwoWayPredictor:
    def __init__(self, mix, x0, xf, mask):
        self.mix = mix
        self.x0 = x0
        self.xf = xf
        self.mask = mask
        self.manifold = mix.manifold
        self.fsde = mix.bridge(xf)
        self.bsde = mix.rev().bridge(x0)

    def update_fn(self, rng, x, t, dt):
        z = self.mix.manifold.random_normal_tangent(
            state=rng, base_point=x, n_samples=x.shape[0]
        )[1].reshape(x.shape[0], -1)
        fdrift, fdiff = self.fsde.coefficients(x, t)
        bdrift, bdiff = self.bsde.coefficients(x, t)

        drift = jnp.einsum("...i,...->...i", fdrift, self.mask) + \
                jnp.einsum("...i,...->...i", bdrift, ~self.mask)
        diffusion = fdiff * self.mask + bdiff * ~self.mask

        tangent_vector = jnp.einsum(
            "...,...i,...->...i", 
            diffusion, z, jnp.abs(jnp.sqrt(dt))
        )
        tangent_vector = tangent_vector + jnp.einsum("...i,...->...i", drift, dt)
        x = self.manifold.exp(tangent_vec=tangent_vector, base_point=x)
        return x, x


def get_twoway_sampler(mix, N=10, eps=1.0e-3,): 
    def sampler(rng, x0, xf, t):
        t_mask = t < 0.5
        predictor = EulerMaruyamaTwoWayPredictor(mix, x0, xf, t_mask)
        x = jnp.einsum("...i,...->...i", x0, t_mask) + \
            jnp.einsum("...i,...->...i", xf, ~t_mask)

        ts = t * t_mask + (1.-t) * ~t_mask
        timesteps = jnp.linspace(start=mix.t0, stop=ts, num=N, endpoint=True)
        dt = (ts - mix.t0) / N

        def loop_body(i, val):
            rng, x, x_mean = val
            t = timesteps[i]
            rng, step_rng = jax.random.split(rng)
            x, x_mean = predictor.update_fn(step_rng, x, t, dt)
            return rng, x, x_mean

        _, x, x_mean = jax.lax.fori_loop(0, N, loop_body, (rng, x, x))
        return x_mean
    return sampler
