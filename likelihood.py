import jax.numpy as jnp
import numpy as np

from util.ode import odeint
from util.proj_integrator import projx_integrator_return_last
from util.utils import div_noise, get_pode_drift, get_riemannian_div_fn


class Flow:
    def __init__(self, mix, rtol, atol, **kwargs):
        self.mix = mix
        self.test_ode_kwargs = dict(rtol=rtol, atol=atol)
        self.eps = kwargs.get("eps", 1e-3)
        self.hutchinson_type = kwargs.get('hutchinson_type', 'None')
        self.manifold = mix.manifold
        self.solver = kwargs.get("method", 'dopri5')

    def get_forward(self, modelf_w_dicts, modelb_w_dicts):
        modelf, paramsf, statesf = modelf_w_dicts
        modelb, paramsb, statesb = modelb_w_dicts

        def forward(data, rng=None):
            shape = data.shape
            epsilon = div_noise(rng, shape, self.hutchinson_type)
            if self.solver=='euler':
                ts = jnp.linspace(
                    start=self.mix.t0, 
                    stop=self.mix.tf - self.eps, 
                    num=1000, endpoint=True
                )
            else:
                ts = jnp.array([self.mix.t0, self.mix.tf - self.eps])

            def ode_func(y, t, paramsf, statesf, paramsb, statesb):
                sample = y[:, :-1]
                vec_t = jnp.ones((sample.shape[0],)) * t

                drift_fn = get_pode_drift(
                    modelf_w_dicts=(modelf, paramsf, statesf), 
                    modelb_w_dicts=(modelb, paramsb, statesb), 
                    mix=self.mix
                )

                drift = drift_fn(sample, vec_t)
                div_fn = get_riemannian_div_fn(
                    drift_fn, 
                    self.hutchinson_type, 
                    self.manifold
                )
                logp_grad = div_fn(sample, vec_t, epsilon).reshape([shape[0], 1])
                return jnp.concatenate([drift, logp_grad], axis=1)

            data = data.reshape(shape[0], -1)
            init = jnp.concatenate([data, np.zeros((shape[0], 1))], axis=1)
            if self.solver == 'euler':
                y = projx_integrator_return_last(
                    self.mix.manifold, 
                    ode_func, init, ts,
                    paramsf, statesf, paramsb, statesb
                )
                nfe = 1000
                z = y[:, :-1].reshape(shape)
                delta_logp = y[:, -1]
            else:
                y, nfe = odeint(
                    ode_func, init, ts, 
                    paramsf, statesf, paramsb, statesb, 
                    **self.test_ode_kwargs
                )
                z = y[-1, ..., :-1].reshape(shape)
                delta_logp = y[-1, ..., -1]
            return z, delta_logp, nfe
        return forward


class Likelihood:
    def __init__(self, mix, rtol=1.0e-5, atol=1.0e-5, **kwargs):
        self.mix = mix
        self.flow = Flow(mix=mix, rtol=rtol, atol=atol, **kwargs)

    def get_log_prob(self, modelf_w_dicts, modelb_w_dicts):
        def log_prob(y, rng=None):
            flow = self.flow.get_forward(modelf_w_dicts, modelb_w_dicts)
            z, inv_logdets, nfe = flow(y, rng=rng) 
            log_prob = self.mix.prior.log_prob(z).reshape(-1)
            log_prob += inv_logdets
            return jnp.clip(log_prob, -1e38, 1e38), nfe
        return log_prob