from .product_manifold import ProductSameManifold
from .hypersphere import Hypersphere


class Torus(ProductSameManifold):
    def __init__(self, dim, **kwargs):
        super(Torus, self).__init__(Hypersphere(1), dim, **kwargs)


if __name__ == "__main__":
    import jax

    torus = Torus(3)
    rng = jax.random.PRNGKey(0)
    rng, next_rng = jax.random.split(rng)
    K = 8
    x = torus.random_uniform(next_rng, K)
    print(x.shape)
    print(torus.belongs(x).all())
    x = torus.projection(x)
    print(torus.belongs(x).all())
    rng, next_rng = jax.random.split(rng)
    _, v = torus.random_normal_tangent(next_rng, x, n_samples=1)
    print(v.shape)
    print(torus.is_tangent(v, x).all())
    print(torus.metric.exp(v, x).shape)
    print(torus.metric.log(v, x).shape)
    print(torus._log_heat_kernel(x, torus.metric.log(v, x), 0.1, 10).shape)
