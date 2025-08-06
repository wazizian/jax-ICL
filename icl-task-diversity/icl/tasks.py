import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import Array

from icl.models import Model, get_model

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


Sampler = Callable[[int], tuple[Array, Array, Array, Array]]


def get_task_name(task: "Task") -> str:
    """
    Get the task type name for evaluation logging.
    
    Returns:
        "Pretrain" for tasks with fresh random task vectors (n_tasks=0)  
        "Latent" for tasks with fixed task pool (n_tasks>0)
    """
    return "Pretrain" if task.name.endswith("(0)") else "Latent"


########################################################################################################################
# Noisy Linear Regression                                                                                              #
########################################################################################################################

"""
Noisy Linear Regression Task for In-Context Learning

This implements a noisy linear regression task y = w^T x + ε where:
- x: input data points (n_dims dimensional)  
- w: task vector (n_dims dimensional)
- ε: Gaussian noise
- y: noisy target values

The task supports two evaluation modes based on task distribution:

**Latent Tasks** (n_tasks > 0):
- Uses a fixed pool of pre-generated task vectors
- Tasks are sampled from this finite pool during training/evaluation
- Model can learn the latent structure of this specific task distribution
- Better for studying how models specialize on repeated task patterns
- Task name ends with the pool size, e.g., "NoisyLinReg(16)"

**Pretrain Tasks** (n_tasks = 0):  
- Generates fresh task vectors from Gaussian distribution each time
- Mimics the diverse task distribution seen during pretraining
- More challenging as model must generalize to completely novel tasks
- Better for studying few-shot learning on unseen tasks
- Task name is "NoisyLinReg(0)"

The task also supports two data sampling modes:

**Fixed Data Pool** (n_data > 0):
- Uses a fixed pool of pre-generated data points
- Data points are sampled from this finite pool during training/evaluation
- Allows studying performance on repeated data patterns

**Fresh Data Sampling** (n_data = 0):
- Generates fresh data points from Gaussian distribution each time
- Each batch contains completely novel data points
- Default behavior for maximum data diversity

Evaluation compares Transformer performance against:
- Ground truth (noise-free predictions)
- Ridge regression baseline (optimal linear predictor given noise/task scales)
"""


@dataclasses.dataclass
class NoisyLinearRegression:
    n_tasks: int
    n_data: int
    n_dims: int
    n_points: int
    batch_size: int
    data_seed: int
    task_seed: int
    noise_seed: int
    data_scale: float
    task_scale: float
    noise_scale: float
    dtype: Any

    def __post_init__(self):
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        self.task_pool = self.generate_task_pool() if self.n_tasks > 0 else None
        self.data_pool = self.generate_data_pool() if self.n_data > 0 else None

    @property
    def name(self) -> str:
        return f"NoisyLinReg({self.n_tasks})"

    @classmethod
    def from_task_pool(cls, task_pool: Array, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        return task

    def generate_task_pool(self) -> Array:
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.n_dims, 1
        tasks = jax.random.normal(key, shape, self.dtype) * self.task_scale
        return tasks

    def generate_data_pool(self) -> Array:
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        return data

    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            data = self.data_pool[idxs]
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        return data

    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
            tasks = self.task_pool[idxs]
        else:
            shape = self.batch_size, self.n_dims, 1
            tasks = jax.random.normal(key, shape, self.dtype) * self.task_scale
        return tasks

    def evaluate(self, data: Array, tasks: Array, step: int) -> Array:
        targets = (data @ tasks)[:, :, 0]
        key = jax.random.fold_in(self.noise_key, step)
        noise = jax.random.normal(key, targets.shape, self.dtype) * self.noise_scale
        return targets + noise

    def generate_attention_mask(self) -> Array:
        """Generate attention mask for the sequence."""
        seq_len = 2 * self.n_points  # Interleaved (x, y) pairs
        mask = jnp.tril(jnp.ones((seq_len, seq_len))).astype(bool)
        return mask

    def sample_batch(self, step: int) -> tuple[Array, Array, Array, Array]:
        data, tasks = self.sample_data(step), self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)
        attention_mask = self.generate_attention_mask()
        return data, tasks, targets, attention_mask

    @staticmethod
    def evaluate_oracle(data: Array, tasks: Array) -> Array:
        targets = (data @ tasks)[:, :, 0]
        return targets

    def get_default_eval_tasks(
        self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, **kwargs
    ) -> list["NoisyLinearRegression"]:
        del kwargs
        assert task_seed != self.task_seed
        assert data_seed != self.data_seed
        assert noise_seed != self.noise_seed
        config = dataclasses.asdict(self)
        config["batch_size"] = batch_size
        config["task_seed"] = task_seed
        config["data_seed"] = data_seed
        config["noise_seed"] = noise_seed
        config["n_tasks"] = 0
        config["n_data"] = 0
        eval_tasks = [self.__class__(**config)]
        if self.n_tasks > 0:
            config["n_tasks"] = self.n_tasks
            eval_tasks.append(NoisyLinearRegression.from_task_pool(**config, task_pool=self.task_pool.copy()))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        models = [get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
        return models


########################################################################################################################
# Get Task                                                                                                             #
########################################################################################################################

Task = NoisyLinearRegression


def get_task(name: str, **kwargs) -> Task:
    tasks = {"noisy_linear_regression": NoisyLinearRegression}
    return tasks[name](**kwargs)
