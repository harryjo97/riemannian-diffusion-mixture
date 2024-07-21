from dataclasses import dataclass
import jax.nn as jnn
import jax.numpy as jnp
import haiku as hk

from util.registry import register_category, register_model

get_activation, register_activation = register_category("activation")

register_activation(jnn.elu, name="elu")
register_activation(jnn.relu, name="relu")
register_activation(jnn.swish, name="swish")
register_activation(jnp.sin, name='sin')


@register_model
@dataclass
class MLP:
    hidden_shapes: list
    output_shape: list
    act: str
    bias: bool = True

    def __call__(self, x):
        for hs in self.hidden_shapes:
            x = hk.Linear(output_size=hs, with_bias=self.bias)(x)
            x = get_activation(self.act)(x)
        x = hk.Linear(output_size=self.output_shape)(x)
        return x


@dataclass
class ScoreNetwork(hk.Module):
    def __init__(self, output_shape, hidden_dim, num_layers, act, **kwargs):
        super().__init__()

        hidden_shapes = kwargs.get('hidden_shapes', None)
        if hidden_shapes is None:
            hidden_shapes = [hidden_dim] * num_layers
        self._layer = MLP(hidden_shapes=hidden_shapes, output_shape=output_shape, act=act)

    def __call__(self, x, t):
        if len(t.shape) == len(x.shape) - 1:
            raise NotImplementedError(f'ScoreNetwork t shape wrong!')
        return self._layer(jnp.concatenate([x, t], axis=-1))