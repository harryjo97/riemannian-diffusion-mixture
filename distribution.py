import jax
import numpy as np
from jax import numpy as jnp
from jax.scipy.special import logsumexp
import geomstats.backend as gs
from distrax import MultivariateNormalDiag

from geomstats.geometry.special_orthogonal import _SpecialOrthogonalMatrices, \
                                                    _SpecialOrthogonal3Vectors

class UniformDistribution:
    """Uniform density on compact manifold"""

    def __init__(self, manifold):
        self.manifold = manifold

    def sample(self, rng, shape):
        return self.manifold.random_uniform(state=rng, n_samples=shape[0])

    def log_prob(self, z):
        return -np.ones([z.shape[0]]) * self.manifold.log_volume


class Wrapped:
    """Wrapped normal density on compact manifold"""

    def __init__(self, scale, batch_dims, manifold, mean_type, seed=0, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng

        if mean_type == 'random':
            self.mean = manifold.random_uniform(state=next_rng, n_samples=1)
        elif mean_type == 'hyperbolic':
            self.mean = jnp.expand_dims(self.manifold.identity, axis=0)
        elif mean_type == 'mixture':
            self.mean = kwargs['mean']
        else:
            raise NotImplementedError(f'mean_type: {mean_type} not implemented.')

        self.scale = jnp.ones((self.mean.shape)) * scale if isinstance(scale, float) \
                    else jnp.array(scale)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample(self.rng, self.batch_dims)

    def sample(self, rng, n_samples):
        if not isinstance(n_samples, int):
            n_samples = n_samples[0]
        mean = self.mean
        scale = self.scale

        tangent_vec = self.manifold.random_normal_tangent(
            rng, mean, n_samples
        )[1]
        tangent_vec = scale * tangent_vec

        samples = self.manifold.exp(tangent_vec, mean)
        return samples

    # Used for SO3 and hyperbolic
    def log_prob(self, samples):
        tangent_vec = self.manifold.metric.log(samples, self.mean)
        tangent_vec = self.manifold.metric.transpback0(self.mean, tangent_vec)
        zero = jnp.zeros((self.manifold.dim))
        # TODO: to refactor axis contenation / removal
        if self.scale.shape[-1] == self.manifold.dim:  # poincare
            scale = self.scale
        else:  # hyperboloid
            scale = self.scale[..., 1:]
        norm_pdf = MultivariateNormalDiag(zero, scale).log_prob(tangent_vec)
        logdetexp = self.manifold.metric.logdetexp(self.mean, samples)
        return norm_pdf - logdetexp


class WrappedMixture:
    """Wrapped normal mixture density on compact manifold"""

    def __init__(self, scale, batch_dims, manifold, mean_type, seed, rng=None, **kwargs):
        self.batch_dims = batch_dims
        self.manifold = manifold
        rng = jax.random.PRNGKey(seed)
        rng, next_rng = jax.random.split(rng)
        self.rng = rng

        if mean_type == 'random':
            self.mean = manifold.random_uniform(state=next_rng, n_samples=4)
        elif mean_type == 'so3':
            assert isinstance(manifold, _SpecialOrthogonalMatrices)
            means = []
            self.centers = [[0.0, 0.0, 0.0], [0.0, 0.0, np.pi], [np.pi, 0.0, np.pi]] 
            for v in self.centers:
                s = _SpecialOrthogonal3Vectors().matrix_from_tait_bryan_angles(np.array(v))
                means.append(s)
            self.mean = jnp.stack(means)
        elif mean_type == 'poincare_disk':
            self.mean = jnp.array([[-0.8, 0.0],[0.8, 0.0],[0.0, -0.8],[0.0, 0.8]])
        elif mean_type == 'hyperboloid4':
            mean = jnp.array([[-0.4, 0.0],[0.4, 0.0],[0.0, -0.4],[0.0, 0.4]])
            self.mean = self.manifold._ball_to_extrinsic_coordinates(mean)
        elif mean_type == 'hyperboloid6':
            hex = [[0., 2.], [np.sqrt(3), 1.], [np.sqrt(3), -1.], [0., -2.], 
                    [-np.sqrt(3), -1.], [-np.sqrt(3), 1.]]
            mean = jnp.array(hex) * 0.3
            self.mean = self.manifold._ball_to_extrinsic_coordinates(mean)
        elif mean_type == 'test':
            self.mean = kwargs['mean']
        else:
            raise NotImplementedError(f'mean_type: {mean_type} not implemented.')

        self.scale = jnp.ones((self.mean.shape)) * scale if isinstance(scale, float) \
                    else jnp.array(scale)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample(self.rng, self.batch_dims)

    def sample(self, rng, n_samples):
        if not isinstance(n_samples, int):
            n_samples = n_samples[0]
        ks = jnp.arange(self.mean.shape[0])
        self.rng, next_rng = jax.random.split(self.rng)
        _, k = gs.random.choice(state=next_rng, a=ks, n=n_samples)
        mean = self.mean[k]
        scale = self.scale[k]

        tangent_vec = self.manifold.random_normal_tangent(
            next_rng, mean, n_samples
        )[1]
        tangent_vec = tangent_vec * scale

        samples = self.manifold.exp(tangent_vec, mean)
        return samples

    def log_prob(self, samples):
        def component_log_prob(mean, scale):
            dist = Wrapped(scale, self.batch_dims, self.manifold, 
                            'mixture', mean=mean)
            return dist.log_prob(samples)

        component_log_like = jax.vmap(component_log_prob)(self.mean, self.scale)
        b = 1 / self.mean.shape[0] * jnp.ones_like(component_log_like)
        return logsumexp(component_log_like, axis=0, b=b)