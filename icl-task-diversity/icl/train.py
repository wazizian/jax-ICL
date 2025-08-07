import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import orbax.checkpoint as ocp
import logging
from flax import jax_utils
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array
from ml_collections import ConfigDict
from hydra.core.hydra_config import HydraConfig

import icl.utils as u
from icl.evaluate import Preds, get_bsln_preds, get_model_preds, mse
from icl.models import Transformer, get_model
from icl.optim import get_optimizer_and_lr_schedule
from icl.tasks import Sampler, Task, get_task, get_task_name


def initialize(model: Transformer, config: ConfigDict) -> tuple[FrozenDict, Array]:
    params_rng, dropout_rng = jr.split(jr.PRNGKey(config.model.seed))
    dummy_data = jnp.ones((config.task.batch_size, config.model.n_points, config.task.n_dims), dtype=model.dtype)
    dummy_targets = jnp.ones((config.task.batch_size, config.model.n_points), dtype=model.dtype)
    dummy_mask = jnp.ones((config.task.batch_size, 2 * config.model.n_points, 2 * config.model.n_points)).astype(bool)
    variables = jax.jit(model.init)(params_rng, dummy_data, dummy_targets, dummy_mask)
    return variables["params"], dropout_rng


def get_sharded_batch_sampler(task: Task) -> Sampler:
    n_devices = jax.local_device_count()

    def sample_batch(step: int) -> tuple[Array, Array, Array, Array, Array]:
        data, tasks, weights, targets, attention_mask = task.sample_batch(step)
        batch_size = data.shape[0]
        batch_per_device = batch_size // n_devices
        
        # Shard data across devices  
        data = data.reshape(n_devices, batch_per_device, *data.shape[1:])
        tasks = tasks.reshape(n_devices, batch_per_device, *tasks.shape[1:])
        weights = weights.reshape(n_devices, batch_per_device, *weights.shape[1:])
        targets = targets.reshape(n_devices, batch_per_device, *targets.shape[1:])
        
        # Expand attention mask to match batch dimensions
        # From (seq_len, seq_len) to (n_devices, batch_per_device, seq_len, seq_len)
        attention_mask = jnp.broadcast_to(
            attention_mask[None, None, :, :], 
            (n_devices, batch_per_device, *attention_mask.shape)
        )
        
        return data, tasks, weights, targets, attention_mask

    return sample_batch


def train_step(state: TrainState, data: Array, weights: Array, targets: Array, attention_mask: Array, dropout_rng: Array) -> TrainState:
    dropout_rng = jr.fold_in(dropout_rng, state.step + 1)

    def loss_fn(params):
        preds = state.apply_fn({"params": params}, data, targets, attention_mask, training=True, rngs={"dropout": dropout_rng})
        # Compute weighted loss: weights should have shape (batch_size,)
        batch_losses = jnp.square(preds - targets).mean(axis=1)  # Mean over sequence length
        weighted_loss = jnp.sum(batch_losses * weights)
        return weighted_loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(state.params)
    grads = jax.lax.pmean(grads, axis_name="device")
    loss = jax.lax.pmean(loss, axis_name="device")
    state = state.apply_gradients(grads=grads)
    return loss, state


def eval_step(state: TrainState, data: Array, targets: Array, attention_mask: Array) -> Array:
    preds = state.apply_fn({"params": state.params}, data, targets, attention_mask, training=False)
    return preds


def _init_log(bsln_preds: Preds, n_dims: int) -> dict:
    log = {"train/step": [], "train/lr": [], "train/loss": [], "eval/step": []}
    for _task_name, _task_preds in bsln_preds.items():
        log[f"eval/{_task_name}"] = {}
        for _bsln_name, _bsln_preds in _task_preds.items():
            log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"] = []
            if _bsln_name != "True":
                _errs = mse(_bsln_preds, _task_preds["True"]) / n_dims
                log[f"eval/{_task_name}"][f"{_bsln_name} | True"] = _errs.tolist()
    return log


def train(config: ConfigDict) -> None:
    # Setup train experiment with Hydra output directory
    hydra_cfg = HydraConfig.get()
    exp_dir = Path(hydra_cfg.runtime.output_dir)
    exp_name = f"train_{u.get_hash(config)}"
    
    logging.info(f"Train Experiment\nNAME: {exp_name}\nOUTPUT_DIR: {exp_dir}\nCONFIG:\n{config}")
    
    # Validate config 
    assert config.model.n_points == config.task.n_points, "Model n_points must match Task n_points"
    assert config.eval.eval_n_points <= config.task.n_points, "Eval n_points must be less than or equal to Task n_points"

    # Config is already saved by Hydra, but save our version too  
    config_file = exp_dir / "config.json"
    with open(config_file, "w") as f:
        serializable_config = u._convert_for_json(config)
        f.write(json.dumps(serializable_config, indent=2))

    # Model, optimizer and lr schedule
    model = get_model(**config.model, dtype=jnp.dtype(config.dtype))
    logging.info(u.tabulate_model(model, config.task.n_dims, config.model.n_points, config.task.batch_size))
    params, dropout_rng = initialize(model, config)
    tx, lr = get_optimizer_and_lr_schedule(**config.training, params=params)
    logging.info("Initialized Model, Optimizer and LR Schedule")

    # Train state
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    state = jax_utils.replicate(state)
    dropout_rngs = jr.split(dropout_rng, jax.local_device_count())
    logging.info("Initialized TrainState")

    # Data samplers
    train_task = get_task(**config.task, dtype=jnp.dtype(config.dtype))
    j_sample_train_batch = jax.jit(get_sharded_batch_sampler(train_task))
    j_samplers_eval_batch = {
        get_task_name(task): jax.jit(get_sharded_batch_sampler(task))
        for task in train_task.get_default_eval_tasks(**config.eval)
    }
    logging.info("Initialized Data Samplers")

    # Steps
    p_train_step = jax.pmap(train_step, axis_name="device", donate_argnums=0)
    p_eval_step = jax.pmap(eval_step, axis_name="device")
    logging.info("Pmap'd Steps")

    # Evaluate baselines
    logging.info("Evaluate Baselines...")
    bsln_preds = get_bsln_preds(train_task, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size)

    # Loggers
    log = _init_log(bsln_preds, config.task.n_dims)

    # Setup checkpoint manager
    ckpt_mngr = ocp.CheckpointManager(exp_dir)
    
    # Training loop
    logging.info("Start Train Loop")
    train_losses = []
    epoch_size = max(1, config.eval.every)
    last_epoch_time = time.time()
    
    for i in range(1, config.training.total_steps + 1):
        # Train step
        data, _, weights, targets, attention_mask = j_sample_train_batch(i)
        
        loss, state = p_train_step(state, data, weights, targets, attention_mask, dropout_rngs)
        train_losses.append(loss.item())
        log["train/step"].append(i)
        log["train/lr"].append(float(lr(i)))
        log["train/loss"].append(loss.item())

        # Evaluate
        if i % config.eval.every == 0 or i == config.training.total_steps:
            # Log time taken for the last epoch 
            t = time.time() - last_epoch_time

            # Calculate and print average training loss over last epoch
            recent_losses = train_losses[-epoch_size:]
            avg_train_loss = sum(recent_losses) / len(recent_losses)
            
            # Log step and lr
            logging.info(f"Step: {i} [{t:.2f}s] | Train Loss (last {len(recent_losses)} steps): {avg_train_loss:.6f} | LR: {float(lr(i)):.6f}")
            
            # Evaluate model
            eval_preds = get_model_preds(
                state, p_eval_step, j_samplers_eval_batch, config.eval.n_samples, config.eval.batch_size
            )
            # Log and print all evaluation metrics
            log["eval/step"].append(i)
            logging.info("=== Evaluation Metrics ===")
            for _task_name, _task_preds in bsln_preds.items():
                logging.info(f"Task: {_task_name}")
                for _bsln_name, _bsln_preds in _task_preds.items():
                    _errs = mse(eval_preds[_task_name]["Transformer"], _bsln_preds) / config.task.n_dims
                    avg_err = _errs.mean().item()
                    log[f"eval/{_task_name}"][f"Transformer | {_bsln_name}"].append(_errs.tolist())
                    logging.info(f"  Transformer vs {_bsln_name}: {avg_err:.6f}")
            logging.info("=========================")

            # Checkpoint - save to Hydra output directory
            ckpt_mngr.save(i, args=ocp.args.StandardSave(jax_utils.unreplicate(state)))

            # Rest last epoch time
            last_epoch_time = time.time()

    # Save logs to Hydra output directory
    with open(exp_dir / "log.json", "w") as f:
        json.dump(log, f, indent=2)

    # Wrap up
    ckpt_mngr.wait_until_finished()
    jr.normal(jr.PRNGKey(0)).block_until_ready()
    return None
