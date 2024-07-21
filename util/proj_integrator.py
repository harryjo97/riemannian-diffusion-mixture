import jax
import jax.numpy as jnp

def projx_integrator_return_last(
    manifold, odefunc, x0, t, 
    paramsf, statesf, paramsb, statesb
):
    """Has a lower memory cost since this doesn't store intermediate values."""

    def loop_body(i, x):
        t0, t1 = t[i], t[i+1]
        dt = t1 - t0
        vt = odefunc(x, t0, paramsf, statesf, paramsb, statesb)
        x = x + dt * vt
        x = jnp.concatenate([manifold.projection(x[:, :-1]), x[:,-1].reshape(-1, 1)], axis=1)
        return x

    xt = x0
    solution = jax.lax.fori_loop(0, len(t)-1, loop_body, xt)
    return solution


def euler_solver(
    manifold, odefunc, x0, t, 
):

    def loop_body(i, x):
        t0, t1 = t[i], t[i+1]
        dt = t1 - t0
        vt = odefunc(x, t0)
        x = x + dt * vt
        x = manifold.projection(x)
        return x

    xt = x0
    solution = jax.lax.fori_loop(0, len(t)-1, loop_body, xt)
    return solution