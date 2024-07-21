import jax
import optax
from jax import numpy as jnp
import jax.random as random
from jax.tree_util import tree_map

from solver import get_twoway_sampler

def get_mix_loss_fn(
        mix, modelf, modelb, 
        num_steps, reduce_mean=False, 
        eps=1e-3, weight_type='importance'
    ):
    reduce_op = jnp.mean if reduce_mean else \
                lambda *args, **kwargs: jnp.sum(*args, **kwargs)
    sampler = get_twoway_sampler(mix, num_steps)
    Z = mix.importance_cum_weight(mix.tf-eps, eps)

    def weight_fn(t):
        if weight_type=='importance':
            weight = jnp.ones_like(t) * Z
        elif weight_type=='default':
            weight = 1./mix.beta_schedule.beta_t(t)
        else:
            raise NotImplementedError(f'{weight_type} not implemented.')
        return weight

    def loss_fn(rng, paramsf, statesf, paramsb, statesb, x):
        # Forward (prior->data) drift
        predf_fn = mix.get_drift_fn(modelf, paramsf, statesf, return_state=True)
        # Backward (data->prior) drift
        predb_fn = mix.get_drift_fn(modelb, paramsb, statesb, return_state=True)

        rng, step_rng = random.split(rng)
        if 'importance' in weight_type:
            t = mix.sample_importance_weighted_time(
                step_rng, 
                (x.shape[0],), 
                eps
            )
        else:
            t = random.uniform(
                step_rng, 
                (x.shape[0],), 
                minval=mix.t0 + eps, 
                maxval=mix.tf - eps
            )

        rng, step_rng = random.split(rng)
        x0 = mix.prior.sample(step_rng, x.shape)
        
        rng, step_rng = random.split(rng)
        xt = sampler(step_rng, x0, x, t)

        # weight
        weight = weight_fn(t)

        # Forward model loss
        predf, new_model_statef = predf_fn(xt, t, step_rng)
        lossesf = predf - mix.bridge(x).drift(xt, t)
        lossesf = weight * 0.5 * mix.manifold.metric.squared_norm(lossesf, xt)
        lossesf = reduce_op(lossesf.reshape(lossesf.shape[0], -1), axis=-1)

        # Backward model loss
        predb, new_model_stateb = predb_fn(xt, mix.tf-t, step_rng)
        lossesb = predb - mix.rev().bridge(x0).drift(xt, mix.tf-t)
        lossesb = weight * 0.5 * mix.manifold.metric.squared_norm(lossesb, xt)
        lossesb = reduce_op(lossesb.reshape(lossesb.shape[0], -1), axis=-1)

        lossf, lossb = jnp.mean(lossesf), jnp.mean(lossesb)
        loss = lossf + lossb

        return loss, (lossf, lossb, new_model_statef, new_model_stateb)
    return loss_fn


def get_ema_loss_step_fn(
    loss_fn,
    optimizerf,
    optimizerb
):
    """Create a one-step training/evaluation function.
    """

    def step_fn(carry_state, batch):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
          carry_state: A tuple (JAX random state, NamedTuple containing the training state).
          batch: A mini-batch of training/evaluation data.

        Returns:
          new_carry_state: The updated tuple of `carry_state`.
          loss: The average loss value of this state.
        """

        (rng, train_state) = carry_state
        rng, step_rng = jax.random.split(rng)

        grad_fn = jax.value_and_grad(loss_fn, argnums=(1,3), has_aux=True)

        paramsf = train_state.paramsf
        model_statef = train_state.model_statef
        paramsb = train_state.paramsb
        model_stateb = train_state.model_stateb
        
        (loss, (lossf, lossb, new_model_statef, new_model_stateb)), grad = grad_fn(
            step_rng, paramsf, model_statef, paramsb, model_stateb, batch
        )

        updatesf, new_opt_statef = optimizerf.update(
            grad[0], 
            train_state.opt_statef, 
            paramsf
        )
        updatesb, new_opt_stateb = optimizerb.update(
            grad[1], 
            train_state.opt_stateb, 
            paramsb
        )

        new_parmasf = optax.apply_updates(paramsf, updatesf)
        new_parmasb = optax.apply_updates(paramsb, updatesb)

        new_params_emaf = tree_map(
            lambda p_ema, p: p_ema * train_state.ema_rate
            + p * (1.0 - train_state.ema_rate),
            train_state.params_emaf,
            new_parmasf,
        )
        new_params_emab = tree_map(
            lambda p_ema, p: p_ema * train_state.ema_rate
            + p * (1.0 - train_state.ema_rate),
            train_state.params_emab,
            new_parmasb,
        )
        
        step = train_state.step + 1
        new_train_state = train_state._replace(
            step=step,
            opt_statef=new_opt_statef,
            model_statef=new_model_statef,
            paramsf=new_parmasf,
            params_emaf=new_params_emaf,
            opt_stateb=new_opt_stateb,
            model_stateb=new_model_stateb,
            paramsb=new_parmasb,
            params_emab=new_params_emab,
        )

        new_carry_state = (rng, new_train_state)
        return new_carry_state, lossf, lossb
    return step_fn