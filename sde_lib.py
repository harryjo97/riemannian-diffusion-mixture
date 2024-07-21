import abc
import numpy as np
from jax import numpy as jnp
import jax

from distribution import UniformDistribution, Wrapped

class Mixture(abc.ABC):
    def __init__(self, manifold, beta_schedule, prior_type='unif', **kwargs):
        """Base Mixture"""
        super().__init__()
        self.manifold = manifold
        self.beta_schedule = beta_schedule
        self.t0 = beta_schedule.t0
        self.tf = beta_schedule.tf
        self.prior_type = prior_type
        self.kwargs = kwargs

    def time_scale(self, t):
        scale = self.beta_schedule.rescale_t_delta(t, self.tf)
        return self.beta_schedule.beta_t(t) / scale

    def diffusion(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        return jnp.sqrt(beta_t)

    @property
    def prior(self):
        if self.prior_type == 'unif':
            return UniformDistribution(self.manifold)
        elif self.prior_type == 'wrapped':
            return Wrapped(
                scale=self.kwargs['scale'],
                batch_dims=self.kwargs['batch_dims'],
                mean_type=self.kwargs['mean_type'],
                manifold=self.manifold
            )
        elif self.prior_type == 'data':
            # NOTE: should be the actual data distribution,
            # but do not need to be implemented
            return None
        else:
            return None

    def importance_cum_weight(self, t, eps):
        if 'Linear' in self.beta_schedule.__class__.__name__:
            #NOTE: Should use linear beta schedule
            if self.beta_schedule._beta == 0:
                return t / self.beta_schedule.beta_0
            else:
                Z = jnp.log(
                    self.beta_schedule.beta_t(t) / self.beta_schedule.beta_t(self.t0+eps)
                ) 
                return Z / self.beta_schedule._beta
        else:
            raise NotImplementedError(f'BetaSchedule not implemented.')


    def sample_importance_weighted_time(self, rng, shape, eps, steps=100):
        Z = self.importance_cum_weight(self.tf-eps, eps=eps)
        quantile = jax.random.uniform(rng, shape, minval=0, maxval=Z)
        lb = jnp.ones_like(quantile) * (self.t0+eps)
        ub = jnp.ones_like(quantile) * (self.tf-eps)

        def bisection_func(carry, idx):
            lb, ub = carry
            mid = (lb + ub) / 2.
            value = self.importance_cum_weight(mid, eps=eps)
            lb = jnp.where(value <= quantile, mid, lb)
            ub = jnp.where(value <= quantile, ub, mid)
            return (lb, ub), idx

        (lb, ub), _ = jax.lax.scan(bisection_func, (lb, ub), jnp.arange(0, steps))
        return (lb + ub) / 2.


class DiffusionMixture(Mixture):
    def __init__(
        self, manifold, beta_schedule, prior_type='unif', 
        drift_scale=1.0, mix_type='log', 
        **kwargs
    ):
        """Diffusion Mixture"""
        super().__init__(manifold, beta_schedule, prior_type, **kwargs)
        self.drift_scale = drift_scale
        self.mix_type = mix_type

    def bridge(self, dest):
        bparams = {
            'manifold': self.manifold, 
            'beta_schedule': self.beta_schedule, 
            'dest': dest, 
            'drift_scale': self.drift_scale
        }
        if self.mix_type == 'log':
            return BrownianBridge(**bparams, **self.kwargs)
        elif 'spec' in self.mix_type:
            return SpectralBridge(**bparams, **self.kwargs)
        else:
            raise NotImplementedError(f'Bridge type: {self.mix_type} not implemented.')

    def get_drift_fn(self, model, params, states, return_state=False):
        def drift_fn(y, t, rng=None):
            drift, new_state = model.apply(params, states, rng, y=y, t=t)
            if return_state:
                return drift, new_state
            else:
                return drift
        return drift_fn

    def probability_ode(self, driftf, driftb):
        return BackwardProbabilityFlowODE(
            self.manifold, driftf, driftb, self.t0, self.tf
        )

    def rev(self):
        # prior of the reverse should be the data distribution
        return DiffusionMixture(
            self.manifold, 
            self.beta_schedule.reverse(), 
            prior_type='data', 
            drift_scale=self.drift_scale, 
            mix_type=self.mix_type, 
            **self.kwargs
        )

    def approx(self, fdrift_fn, bdrift_fn, use_pode):
        return ApproxMixture(
            self.manifold, 
            self.beta_schedule, 
            self.prior_type, 
            fdrift_fn, 
            bdrift_fn, 
            use_pode, 
            **self.kwargs
        )


class Bridge(abc.ABC):
    def __init__(self, manifold, beta_schedule, dest, drift_scale):
        self.manifold = manifold
        self.beta_schedule = beta_schedule
        self.t0 = beta_schedule.t0
        self.tf = beta_schedule.tf
        self.dest = dest
        self.drift_scale = drift_scale

    def time_scale(self, t):
        scale = self.beta_schedule.rescale_t_delta(t, self.tf)
        return self.beta_schedule.beta_t(t) / scale
    
    # Time-scaled drift
    def drift(self, x, t):
        drift = self.drift_before_scale(x, t)
        coeff = self.time_scale(t) * self.drift_scale
        return jnp.einsum("...i,...->...i", drift, coeff)

    def diffusion(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        return jnp.sqrt(beta_t)

    def coefficients(self, x, t):
        return self.drift(x, t), self.diffusion(x, t)

    @abc.abstractmethod
    def drift_before_scale(self, x, t):
        raise NotImplementedError()


class BrownianBridge(Bridge):
    def __init__(self, manifold, beta_schedule, dest, drift_scale, **kwargs):
        super().__init__(manifold, beta_schedule, dest, drift_scale)

    def drift_before_scale(self, x, t):
        return self.manifold.log(point=self.dest, base_point=x)


class SpectralBridge(Bridge):
    def __init__(self, manifold, beta_schedule, dest, drift_scale, **kwargs):
        super().__init__(manifold, beta_schedule, dest, drift_scale)
        self.wtype = kwargs.get('wtype', 'biharmonic')
        self.tau = kwargs.get('tau', 0.25)
        self.dest_eig = manifold.eig_fn(dest)
        self.weight = self.weighting_fn(manifold.eig_val)

    def weighting_fn(self, z):
        if self.wtype == 'biharmonic':
            return 1./z**2
        elif 'inv' in self.wtype:
            pow = -float(self.wtype.split('_')[-1])
            return jnp.abs(z)**pow 
        elif self.wtype == 'diff':
            return jnp.exp(-2 * self.tau * z)
        else:
            raise NotImplementedError(f'Weighting function {self.wtype} not implemented.')

    def dist(self, z):
        coeff = self.dest_eig - self.manifold.eig_fn(z)
        return (self.weight * coeff**2).sum(-1)

    def drift_before_scale(self, x, t):
        dist, vjp_fn = jax.vjp(lambda y: self.dist(y), x) 
        grad = vjp_fn(jnp.ones_like(dist))[0]
        sqnorm = self.manifold.metric.squared_norm(grad, x).clip(min=1e-20)
        
        # Determine the sign and scale
        drift = -2 * jnp.einsum('...i,...->...i', grad, dist/sqnorm) 
        return drift


# Data -> Prior
class BackwardProbabilityFlowODE:
    def __init__(self, manifold, driftf, driftb, t0, tf):
        self.manifold = manifold
        self.driftf = driftf # Prior -> Data
        self.driftb = driftb # Data -> Prior
        self.t0 = t0
        self.tf = tf

    def coefficients(self, x, t):
        fdrift = self.driftf(x, self.tf-t)
        bdrift = self.driftb(x, t)
        scaled_score_fn = fdrift + bdrift
        ode_drift = bdrift - 0.5 * scaled_score_fn
        return ode_drift, jnp.zeros_like(t)


class ApproxMixture(Mixture):
    def __init__(
        self, manifold, beta_schedule, prior_type='unif', 
        fdrift_fn=None, bdrift_fn=None, use_pode=False, **kwargs
    ):
        """Approximated Diffusion Mixture"""
        super().__init__(manifold, beta_schedule, prior_type, **kwargs)
        self.approx = True
        self.fdrift_fn = fdrift_fn
        self.bdrift_fn = bdrift_fn
        self.use_pode = use_pode

    def diffusion(self, x, t):
        beta_t = self.beta_schedule.beta_t(t)
        return jnp.sqrt(beta_t) if not self.use_pode else jnp.zeros_like(t)

    def drift(self, x, t):
        drift = self.fdrift_fn(x, t)
        if self.use_pode:
            scaled_score_fn = drift + self.bdrift_fn(x, self.tf-t)
            drfit = drift - 0.5 * scaled_score_fn
        return drift

    def coefficients(self, x, t):
        return self.drift(x, t), self.diffusion(x, t)

    def prior_sampling(self, rng, shape):
        return self.prior.sample(rng, shape)
