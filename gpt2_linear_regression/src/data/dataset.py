import jax
import jax.numpy as jnp
import numpy as np
from typing import Iterator, Tuple, Dict, Any
from omegaconf import DictConfig


class LinearRegressionDataset:
    """Dataset that generates linear regression sequences with Gaussian priors."""
    
    def __init__(self, config: DictConfig, split: str = "train"):
        self.config = config.data
        self.split = split
        
        # Set number of sequences based on split
        if split == "train":
            self.num_sequences = self.config.num_sequences
        else:
            self.num_sequences = self.config.num_test_sequences
            
        
    def _sample_weight_vector(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Sample weight vector from Gaussian prior."""
        return jax.random.normal(
            key, 
            shape=(self.config.input_dim,)
        ) * self.config.prior_std + self.config.prior_mean
    
    def _sample_x(self, key: jax.random.PRNGKey, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Sample input vectors x_i uniformly from specified range."""
        return jax.random.uniform(
            key, 
            shape=shape,
            minval=self.config.x_range[0],
            maxval=self.config.x_range[1]
        )
    
    
    def generate_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jnp.ndarray]:
        """Generate a batch of continuous sequences for linear model."""
        keys = jax.random.split(key, batch_size)
        
        inputs_batch = []
        targets_batch = []
        
        for seq_key in keys:
            # Generate continuous sequence data
            key_w, key_x, key_noise = jax.random.split(seq_key, 3)
            
            # Sample weight vector for this sequence
            w = self._sample_weight_vector(key_w)
            
            # Sample input vectors
            x = self._sample_x(
                key_x, 
                (self.config.sequence_length, self.config.input_dim)
            )
            
            # Compute y = w^T x + noise
            y = jnp.dot(x, w)
            if self.config.noise_std > 0:
                noise = jax.random.normal(
                    key_noise, 
                    shape=(self.config.sequence_length,)
                ) * self.config.noise_std
                y = y + noise
            
            # Normalize inputs if requested
            if self.config.normalize_inputs:
                x_mean = jnp.mean(x, axis=0, keepdims=True)
                x_std = jnp.std(x, axis=0, keepdims=True) + 1e-8
                x = (x - x_mean) / x_std
            
            # Create input sequence: concatenate x with previous y values
            # Format: [x_i, y_{i-1}] where y_{-1} = 0
            y_prev = jnp.concatenate([jnp.array([0.0]), y[:-1]])  # Shift targets by 1
            inputs = jnp.concatenate([x, y_prev.reshape(-1, 1)], axis=1)
            
            inputs_batch.append(inputs)
            targets_batch.append(y)
        
        inputs = jnp.stack(inputs_batch)
        targets = jnp.stack(targets_batch)
        
        return {
            "inputs": inputs,
            "targets": targets,
            "attention_mask": jnp.ones_like(targets)
        }
    
    
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """Iterator over batches."""
        key = jax.random.PRNGKey(42 if self.split == "train" else 24)
        
        num_batches = self.num_sequences // self.config.batch_size
        
        for _ in range(num_batches):
            key, batch_key = jax.random.split(key)
            yield self.generate_batch(batch_key, self.config.batch_size)


def create_dataset(config: DictConfig, split: str = "train") -> LinearRegressionDataset:
    """Factory function to create dataset."""
    return LinearRegressionDataset(config, split)