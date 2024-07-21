import os
import socket
import logging
from timeit import default_timer as timer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm # For terminal print

import jax
from jax import numpy as jnp
import optax
import haiku as hk
import math
import numpy as np

from omegaconf import OmegaConf
from hydra.utils import instantiate, get_class

from losses import get_ema_loss_step_fn
from data.split import random_split
from data.tensordataset import DataLoader, TensorDataset
from util.loggers_pl import LoggerCollection
from util.training import TrainState, save, restore
from util.vis import earth_plot, plot_tori, plot_mesh, plot_hyperbolic

log = logging.getLogger(__name__)

def run(cfg):
    def train(train_state, best_val=False):
        best_logp = -200

        loss = instantiate(cfg.loss, mix=mix, modelf=modelf, modelb=modelb)
        train_step_fn = get_ema_loss_step_fn(
            loss, 
            optimizerf=optimizerf, 
            optimizerb=optimizerb
        )
        train_step_fn = jax.jit(train_step_fn)

        rng = train_state.rng
        tbar = tqdm(
            range(train_state.step, cfg.steps),
            total=cfg.steps - train_state.step,
            bar_format="{desc}{bar}{r_bar}",
            mininterval=1,
        )
        train_time = timer()

        total_train_time = 0
        for _ in tbar:
            batch = next(train_ds)
            rng, next_rng = jax.random.split(rng)
            (rng, train_state), lossf, lossb = train_step_fn((next_rng, train_state), batch)
            if jnp.isnan(lossf+lossb).any():
                log.warning("Loss is nan")
                return train_state, best_logp, False

            step = train_state.step
            if step % 100 == 0:
                tbar.set_description(f"F: {lossf:.2f} | B: {lossb:.2f}")
                logger.log_metrics({"train/lossf": lossf}, step)
                logger.log_metrics({"train/lossb": lossb}, step)

            if step > 0 and step % cfg.val_freq == 0:
                logger.log_metrics(
                    {"train/time_per_it": (timer() - train_time) / cfg.val_freq}, 
                    step
                )
                total_train_time += timer() - train_time
                eval_time = timer()

                if cfg.train_val:
                    logp = evaluate(train_state, "val", step)
                    logger.log_metrics({"val/time_per_it": (timer() - eval_time)}, step)

                    if cfg.get('val_test', False):
                        evaluate(train_state, "test", step, after_val=False)

                    if best_val:
                        if logp > best_logp:
                            best_logp = logp
                            save(ckpt_path, train_state)
                    else:
                        save(ckpt_path, train_state)

                if cfg.train_plot and step % cfg.plot_freq == 0:
                    generate_plots(train_state, "val", step=step)
                train_time = timer()
                logger.save()
        logger.log_metrics({"train/total_time": total_train_time}, step)
        return train_state, best_logp, True

    def evaluate(train_state, stage, step, **kwargs):
        try:
            log.info("Running evaluation")
            dataset = eval_ds if stage == "val" else test_ds

            modelf_w_dicts = (modelf, train_state.params_emaf, train_state.model_statef)
            modelb_w_dicts = (modelb, train_state.params_emab, train_state.model_stateb)

            likelihood_fn = likelihood.get_log_prob(modelf_w_dicts, modelb_w_dicts)
            likelihood_fn = jax.jit(likelihood_fn)

            logp, nfe, N, tot = 0.0, 0.0, 0, 0
            if hasattr(dataset, "__len__"):
                for batch in dataset:
                    if len(batch)>0:
                        logp_step, nfe_step = likelihood_fn(batch)
                        logp += logp_step.sum()
                        nfe += nfe_step
                        N += logp_step.shape[0]
            else:
                dataset.batch_dims = [cfg.eval_batch_size]
                num_rounds = round(20000 / cfg.eval_batch_size)
                for i in range(num_rounds):
                    batch = next(dataset)
                    logp_step, nfe_step = likelihood_fn(batch)
                    logp += logp_step.sum()
                    nfe += nfe_step
                    N += logp_step.shape[0]
                    tot += logp_step.shape[0]
                dataset.batch_dims = [cfg.batch_size]

            logp /= N
            nfe /= len(dataset) if hasattr(dataset, "__len__") else num_rounds

            logger.log_metrics({f"{stage}/logp": logp}, step)
            logger.log_metrics({f"{stage}/nfe": nfe}, step)

            with logging_redirect_tqdm():
                if stage == "test" and cfg.best_val and kwargs.get('after_val', True):
                    log.info(f">>> [Epoch {step:06d}] | Val logp={kwargs['best_logp']:.3f} | "
                                f"Test logp={logp:.3f} | nfe: {nfe:.1f}")
                else:
                    log.info(f"[Epoch {step:06d}] {stage} logp: {logp:.3f} | nfe: {nfe:.1f}")
            logger.save()
            return logp
        except:
            with logging_redirect_tqdm():
                log.info('Likelihood computation failed.')
            return -1e+4

    def generate_plots(train_state, stage, step=None):
        try:
            log.info("Generating plots")
            rng = jax.random.PRNGKey(cfg.seed)
            dataset = eval_ds if stage == "eval" else test_ds

            # Generate samples
            modelf_w_dicts = (modelf, train_state.params_emaf, train_state.model_statef)
            modelb_w_dicts = (modelb, train_state.params_emab, train_state.model_stateb)

            likelihood_fn = likelihood.get_log_prob(modelf_w_dicts, modelb_w_dicts)
            log_prob = lambda x: likelihood_fn(x)[0]
            log_prob = jax.jit(log_prob)

            sde = mix.approx(
                mix.get_drift_fn(*modelf_w_dicts),
                mix.rev().get_drift_fn(*modelb_w_dicts),
                cfg.use_pode
            )

            if 'hyperbolic' in cfg.name:
                plt = plot_hyperbolic(test_ds, log_prob)
            else:
                x0 = next(dataset)
                sampler = instantiate(cfg.sampler, sde=sde, N=1000, eps=cfg.eps)
                sampler = jax.jit(sampler)

                NUM_SAMPLES = cfg.get('num_plot_samples', 8192) #8192
                shape = (cfg.sample_batch_size,)
                samples = []
                num_rounds = math.ceil(NUM_SAMPLES / shape[0])
                for i in tqdm(range(num_rounds), position=1, leave=False):
                    rng, next_rng = jax.random.split(rng)
                    x_init = sde.prior.sample(next_rng, shape)
                    samples.append(sampler(rng, x_init))
                samples = jnp.concatenate(samples, axis=0)

                prop_in_M = manifold.belongs(samples, atol=1e-4).mean()
                log.info(f"Prop samples in M = {100 * prop_in_M.item():.1f}%")

                if cfg.name in ['flood', 'fire', 'earthquake', 'volcano']:
                    logp = log_prob(samples)
                    plt = earth_plot(cfg.dataset.name, train_ds, eval_ds, samples, logp)
                elif cfg.name == 'tn':
                    rng, next_rng = jax.random.split(rng)
                    data_samples = eval_ds.sample(next_rng, shape)
                    plt = plot_tori(data_samples, samples)
                elif cfg.name in ['spot50', 'spot100', 'bunny50', 'bunny100']:
                    log_dir = f'logs/version_{logger.version}'
                    save_path = os.path.join(*[run_path, log_dir, 'images'])
                    logprobs = []
                    bs = 1e+4
                    num_rounds = math.ceil(len(manifold.vt)/bs) 
                    for i in tqdm(range(num_rounds), position=1, leave=False):
                        mv = manifold.vt[bs*i:bs*(i+1)] \
                            if i<num_rounds-1 \
                            else manifold.vt[bs*i:]
                        logprobs.append(log_prob(mv))
                    logprobs = np.concatenate(logprobs, axis=0)
                    prob = np.exp(logprobs)
                    plt = plot_mesh(
                        cfg.name, 
                        manifold.vn, manifold.fn, 
                        samples, prob,
                        save_path, step,
                        stage
                    )
                else:
                    raise NotImplementedError(f'Exp: {cfg.name} plot not implemented.')

            if plt is not None:
                logger.log_plot(f"", plt, step)
        except:
            with logging_redirect_tqdm():
                log.info('Plot failed.')


    ### Main
    log.info(cfg)
    log.info(f"Jax devices: {jax.devices()}")
    log.info("Stage : Start")
    run_path = os.getcwd()
    log.info(f"run_path: {run_path}")
    log.info(f"hostname: {socket.gethostname()}")
    ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)

    loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logger.values()]
    logger = LoggerCollection(loggers)
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    rng = jax.random.PRNGKey(cfg.seed)

    log.info("Stage : Instantiate dataset")
    rng, next_rng = jax.random.split(rng)
    dataset = instantiate(cfg.dataset, rng=next_rng)

    if isinstance(dataset, TensorDataset):
        # split and wrapp dataset into dataloaders
        train_ds, eval_ds, test_ds = random_split(
            dataset, lengths=cfg.splits, rng=next_rng
        )
        train_ds, eval_ds, test_ds = (
            DataLoader(train_ds, batch_dims=cfg.batch_size, rng=next_rng, shuffle=True),
            DataLoader(eval_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
            DataLoader(test_ds, batch_dims=cfg.eval_batch_size, rng=next_rng),
        )
        log.info(
            f"Train size: {len(train_ds.dataset)}. Val size: {len(eval_ds.dataset)}. Test size: {len(test_ds.dataset)}"
        )
    else:
        train_ds, eval_ds, test_ds = dataset, dataset, dataset

    manifold = dataset.manifold

    log.info("Stage : Instantiate mixture")
    beta_schedule = instantiate(cfg.beta_schedule)
    mix = instantiate(cfg.mix, manifold=manifold, beta_schedule=beta_schedule)
    likelihood = instantiate(cfg.likelihood, mix=mix)

    log.info("Stage : Instantiate model")

    modelf_cfg = cfg.get('model', cfg.get('modelf'))
    modelb_cfg = cfg.get('model', cfg.get('modelb'))

    def fmodel(y, t):
        output_shape = get_class(cfg.generator._target_).output_shape(manifold)
        drift_fn = instantiate(
            cfg.generator,
            modelf_cfg,
            output_shape,
            manifold
        )
        return drift_fn(y, jnp.expand_dims(t.reshape(-1), -1))

    def bmodel(y, t):
        output_shape = get_class(cfg.generator._target_).output_shape(manifold)
        drift_fn = instantiate(
            cfg.generator,
            modelb_cfg,
            output_shape,
            manifold
        )
        return drift_fn(y, jnp.expand_dims(t.reshape(-1), -1))

    modelf = hk.transform_with_state(fmodel)
    modelb = hk.transform_with_state(bmodel)

    rng, next_rng = jax.random.split(rng)
    t = jnp.zeros((cfg.batch_size, 1))
    data= next(train_ds)

    paramsf, statef = modelf.init(rng=next_rng, y=data, t=t)
    paramsb, stateb = modelb.init(rng=next_rng, y=data, t=t)

    log.info("Stage : Instantiate scheduler/optimizer")
    schedule_fnf = instantiate(cfg.scheduler)
    optimizerf = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fnf))
    opt_statef = optimizerf.init(paramsf)
    schedule_fnb = instantiate(cfg.scheduler)
    optimizerb = optax.chain(instantiate(cfg.optim), optax.scale_by_schedule(schedule_fnb))
    opt_stateb = optimizerb.init(paramsb)

    if cfg.resume or cfg.mode == "test":  
        # if resume or evaluate
        train_state = restore(ckpt_path)
        best_logp = -1e+4
    else:
        rng, next_rng = jax.random.split(rng)
        train_state = TrainState(
            opt_statef=opt_statef,
            model_statef=statef,
            paramsf=paramsf,
            params_emaf=paramsf,
            opt_stateb=opt_stateb,
            model_stateb=stateb,
            paramsb=paramsb,
            params_emab=paramsb,
            step=0,
            ema_rate=cfg.ema_rate,
            rng=next_rng,
        )
        save(ckpt_path, train_state)

    if cfg.mode == "train" or cfg.mode == "all":
        if train_state.step == 0 and cfg.test_plot:
            generate_plots(train_state, "test", step=-1)

        log.info("Stage : Training")
        train_state, best_logp, success = train(train_state, cfg.best_val)

    if cfg.mode == "test" or (cfg.mode == "all" and success):
        train_state = restore(ckpt_path)
        
        log.info("Stage : Test")
        if cfg.test_test:
            evaluate(train_state, "test", step=train_state.step, best_logp=best_logp)
        if cfg.test_plot:
            generate_plots(train_state, "test", step=train_state.step)
        success = True
    logger.save()
    logger.finalize("success" if success else "failure")