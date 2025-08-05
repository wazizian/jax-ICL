import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.checkpoints import save_checkpoint, restore_checkpoint
from flax import jax_utils
import numpy as np
from typing import Dict, Any, Tuple, Optional
from omegaconf import DictConfig
import logging
import time
import os
from functools import partial

from ..models.gpt2_model import create_model
from ..data.dataset import create_dataset


class TrainState(train_state.TrainState):
    """Extended train state with additional metrics."""
    dropout_rng: jax.random.PRNGKey


def compute_loss(predictions: jnp.ndarray, targets: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
    """Compute MSE loss for continuous outputs."""
    # Flatten for loss computation
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    mask_flat = attention_mask.reshape(-1)
    
    # Compute MSE loss
    loss = (pred_flat - target_flat) ** 2
    
    # Apply mask and compute mean
    loss = loss * mask_flat
    total_loss = jnp.sum(loss)
    total_mask = jnp.sum(mask_flat)
    
    # Avoid division by zero
    loss = jnp.where(total_mask > 0, total_loss / total_mask, 0.0)
    
    return loss


def compute_metrics(predictions: jnp.ndarray, targets: jnp.ndarray, attention_mask: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Compute training metrics for continuous outputs."""
    loss = compute_loss(predictions, targets, attention_mask)
    
    # Compute MAE (Mean Absolute Error)
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    mask_flat = attention_mask.reshape(-1)
    
    mae = jnp.abs(pred_flat - target_flat) * mask_flat
    mae = jnp.sum(mae) / jnp.sum(mask_flat)
    
    # Compute R² coefficient
    target_mean = jnp.sum(target_flat * mask_flat) / jnp.sum(mask_flat)
    ss_tot = jnp.sum(((target_flat - target_mean) ** 2) * mask_flat)
    ss_res = jnp.sum(((target_flat - pred_flat) ** 2) * mask_flat) 
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    return {
        'loss': loss,
        'mae': mae,
        'r2': r2
    }


class Trainer:
    """Trainer class for GPT2 model."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.rng = jax.random.PRNGKey(config.seed)
        
        # Initialize linear model
        self.model = create_model(config)
        
        # Initialize datasets
        self.train_dataset = create_dataset(config, split="train")
        self.eval_dataset = create_dataset(config, split="test")
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize train state
        self.state = self._create_train_state()
        
        # Compile training and evaluation steps
        if config.training.compile_train_step:
            self.train_step = jax.jit(self._train_step)
        else:
            self.train_step = self._train_step
            
        if config.training.compile_eval_step:
            self.eval_step = jax.jit(self._eval_step)
        else:
            self.eval_step = self._eval_step
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.log_level))
        self.logger = logging.getLogger(__name__)
        
        # Setup checkpointing
        if config.training.save_checkpoints:
            os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with warmup and weight decay."""
        # Linear warmup
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.training.learning_rate,
            transition_steps=self.config.training.warmup_steps
        )
        
        # Constant learning rate after warmup
        constant_schedule = optax.constant_schedule(self.config.training.learning_rate)
        
        # Combine schedules
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, constant_schedule],
            boundaries=[self.config.training.warmup_steps]
        )
        
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.training.max_grad_norm),
            optax.adamw(
                learning_rate=schedule,
                b1=self.config.training.beta1,
                b2=self.config.training.beta2,
                eps=self.config.training.epsilon,
                weight_decay=self.config.training.weight_decay
            )
        )
        
        return optimizer
    
    def _create_train_state(self) -> TrainState:
        """Create initial training state."""
        # Initialize model parameters
        self.rng, init_rng = jax.random.split(self.rng)
        
        # Create dummy input for initialization: (batch_size, seq_len, input_dim)
        input_dim = self.config.data.input_dim + 1  # +1 for target value
        dummy_input = jnp.ones((1, 10, input_dim), dtype=jnp.float32)
        variables = self.model.init(init_rng, dummy_input)
        
        # Create train state
        self.rng, dropout_rng = jax.random.split(self.rng)
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer,
            dropout_rng=dropout_rng
        )
        
        return state
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state: TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Single training step."""
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        
        def loss_fn(params):
            predictions = state.apply_fn(
                {'params': params},
                batch['inputs'],
                attention_mask=batch['attention_mask'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            loss = compute_loss(predictions, batch['targets'], batch['attention_mask'])
            return loss, predictions
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, predictions), grads = grad_fn(state.params)
        
        # Update parameters
        state = state.apply_gradients(grads=grads, dropout_rng=new_dropout_rng)
        
        # Compute metrics
        metrics = compute_metrics(predictions, batch['targets'], batch['attention_mask'])
        
        return state, metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def _eval_step(self, state: TrainState, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Single evaluation step."""
        predictions = state.apply_fn(
            {'params': state.params},
            batch['inputs'],
            attention_mask=batch['attention_mask'],
            deterministic=True
        )
        
        metrics = compute_metrics(predictions, batch['targets'], batch['attention_mask'])
        return metrics
    
    def evaluate(self, num_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on test set."""
        eval_metrics = []
        
        eval_iter = iter(self.eval_dataset)
        batch_count = 0
        
        for batch in eval_iter:
            if num_batches is not None and batch_count >= num_batches:
                break
                
            metrics = self.eval_step(self.state, batch)
            eval_metrics.append(metrics)
            batch_count += 1
        
        # Average metrics
        avg_metrics = {}
        for key in eval_metrics[0].keys():
            avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in eval_metrics])))
        
        return avg_metrics
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        if self.config.training.save_checkpoints:
            save_checkpoint(
                ckpt_dir=self.config.training.checkpoint_dir,
                target=self.state,
                step=step,
                keep=self.config.training.max_checkpoints_to_keep
            )
    
    def load_checkpoint(self, step: Optional[int] = None):
        """Load model checkpoint."""
        if self.config.training.save_checkpoints:
            self.state = restore_checkpoint(
                ckpt_dir=self.config.training.checkpoint_dir,
                target=self.state,
                step=step
            )
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        step = 0
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.training.num_epochs}")
            
            train_iter = iter(self.train_dataset)
            epoch_metrics = []
            
            for batch in train_iter:
                # Training step
                self.state, metrics = self.train_step(self.state, batch)
                epoch_metrics.append(metrics)
                step += 1
                
                # Logging
                if step % self.config.training.log_every_n_steps == 0:
                    avg_metrics = {}
                    for key in metrics.keys():
                        avg_metrics[key] = float(jnp.mean(jnp.array([m[key] for m in epoch_metrics[-self.config.training.log_every_n_steps:]])))
                    
                    self.logger.info(
                        f"Step {step}: " + 
                        ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
                    )
                
                # Evaluation
                if step % self.config.training.eval_every_n_steps == 0:
                    eval_metrics = self.evaluate(num_batches=10)  # Quick evaluation
                    self.logger.info(
                        f"Eval at step {step}: " +
                        ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
                    )
                
                # Checkpointing
                if step % self.config.training.save_every_n_steps == 0:
                    self.save_checkpoint(step)
                    self.logger.info(f"Saved checkpoint at step {step}")
            
            # End of epoch evaluation
            eval_metrics = self.evaluate()
            self.logger.info(
                f"End of epoch {epoch + 1} evaluation: " +
                ", ".join([f"{k}: {v:.4f}" for k, v in eval_metrics.items()])
            )
        
        # Final checkpoint
        self.save_checkpoint(step)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        return self.state