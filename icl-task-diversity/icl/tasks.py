import dataclasses
from typing import Any, Callable, List

import jax
import jax.numpy as jnp
from jax import Array, tree_util
from functools import partial
import chex

from icl.models import Model, get_model

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


Sampler = Callable[[int], tuple[Array, Array, Array, Array, Array]]


def get_task_name(task: "Task") -> str:
    return task.name 

@partial(jax.jit, static_argnames=("shape", "dtype"))
def sample_truncated_normal(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        clip: float,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    def cond_fun(val):
        _, x = val
        return jnp.any(jnp.abs(x) > clip)

    def body_fun(val):
        key, x = val
        key, subkey = jax.random.split(key)
        new_sample = jax.random.normal(subkey, shape=shape, dtype=dtype) * scale + loc
        new_x = jax.lax.select(jnp.abs(x) > clip, new_sample, x)
        return key, new_x

    key, subkey = jax.random.split(key)
    init_x = jax.random.normal(subkey, shape=shape, dtype=dtype) * scale + loc
    init_val = (key, init_x)
    _, final_x = jax.lax.while_loop(cond_fun, body_fun, init_val)
    return final_x

def sample_multivariate_gaussian(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        clip: float | None,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    if clip is not None:
        return jax.random.truncated_normal(
            key,
            lower=(-clip - loc) / scale,
            upper=(clip - loc) / scale,
            shape=shape,
            dtype=dtype
            ) * scale + loc
    else:
        return jax.random.normal(key, shape=shape, dtype=dtype) * scale + loc


def sample_student_t(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        df: float,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    """Sample from Student-t distribution with location and scale."""
    adjusted_scale = scale * jnp.sqrt((df - 2) / df)  # Adjust scale for variance
    return jax.random.t(key, df, shape=shape, dtype=dtype) * adjusted_scale + loc


def sample_distrib(
        key: jax.random.PRNGKey,
        loc: Array,
        scale: Array,
        clip: float | None,
        distrib_name: str,
        distrib_param: float | None,
        shape: tuple[int, ...],
        dtype: Any = jnp.float32
        ) -> Array:
    """Dispatch to appropriate sampling function based on distribution name."""
    if distrib_name == "normal":
        return sample_multivariate_gaussian(key, loc, scale, clip, shape, dtype)
    elif distrib_name == "student":
        if clip is not None:
            raise NotImplementedError("Student-t distribution with clipping not implemented")
        if distrib_param is None:
            raise ValueError("distrib_param (degrees of freedom) must be specified for student-t distribution")
        return sample_student_t(key, loc, scale, distrib_param, shape, dtype)
    else:
        raise ValueError(f"Unknown distribution name: {distrib_name}")

#@partial(jax.jit, static_argnames=("clip",))
def task_log_weights(
        tasks: Array,
        loc: float,
        scale: float,
        clip: float | None,
        distrib_name: str,
        distrib_param: float | None,
        use_weights: bool = False,
        reduce_axis: int = -1
        ) -> Array:
    if not use_weights:
        return jnp.zeros_like(tasks).sum(axis=reduce_axis)
    
    if distrib_name == "normal":
        if clip is None:
            log_weights = jax.scipy.stats.norm.logpdf(tasks, loc=loc, scale=scale)
        else:
            log_weights = jax.scipy.stats.truncnorm.logpdf(tasks, -clip, clip, loc=loc, scale=scale)
    elif distrib_name == "student":
        if clip is not None:
            raise NotImplementedError("Student-t distribution with clipping not implemented")
        if distrib_param is None:
            raise ValueError("distrib_param (degrees of freedom) must be specified for student-t distribution")
        # Student-t logpdf: loc and scale parameters
        assert distrib_param > 2, "Degrees of freedom must be greater than 2 for Student-t distribution"
        # Match the scale so that variance stays constant as distrib_param changes
        adjusted_scale = scale * jnp.sqrt((distrib_param - 2) / distrib_param)
        standardized = (tasks - loc) / adjusted_scale
        log_weights = jax.scipy.stats.t.logpdf(standardized, df=distrib_param) - jnp.log(adjusted_scale)
    else:
        raise ValueError(f"Unknown distribution name: {distrib_name}")
    
    return - jnp.sum(log_weights, axis=reduce_axis)  # IMPORTANT: minus

def task_weights_trunc_norm_factor(
        loc: float,
        scale: float,
        clip: float | None,
        use_weights: bool,
        n_dims: int
        ) -> float:
    if not use_weights:
        return 1.0
    # Compute integral of exp(-log_weights) over R^d
    assert clip is not None, "clip must be specified to compute normalization factor"
    x = jnp.linspace(-clip, clip, 1000)
    Z_1d = jax.numpy.trapezoid(
            jnp.exp(-jax.scipy.stats.truncnorm.logpdf(x, -clip, clip, loc=loc, scale=scale)),
            x,
            )
    jax.debug.print("Trunc norm factor 1d: {}", Z_1d)
    return Z_1d ** n_dims


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
    task_center: float | None = None
    n_max_points: int | None = None  # Optional, used for padding in some models
    clip: float | None = None  # Optional, clip task vectors to [-clip, clip]^d
    name: str | None = None  # Optional, can be set to override default name
    eval_ridge: bool = True  # Optional, whether to include Ridge baseline in evaluation
    use_weights: bool = False  # Optional, whether to use task importance weights
    distrib_name: str = "normal"  # Distribution name: "normal" or "student"
    distrib_param: float | None = None  # Distribution parameter (degrees of freedom for student-t)

    def __post_init__(self):
        # Validation
        if self.distrib_name == "student" and self.clip is not None:
            raise NotImplementedError("Student-t distribution with clipping not implemented")
            
        self.data_key = jax.random.PRNGKey(self.data_seed)
        self.task_key = jax.random.PRNGKey(self.task_seed)
        self.noise_key = jax.random.PRNGKey(self.noise_seed)
        self.n_max_points = self.n_points if self.n_max_points is None else self.n_max_points
        self.task_center = 0.0 if self.task_center is None else self.task_center
        task_pool, weights = self.generate_task_pool() if self.n_tasks > 0 else (None, None)
        self.task_pool = task_pool
        self.weights = weights
        self.data_pool = self.generate_data_pool() if self.n_data > 0 else None
        self.name = f"NoisyLinReg({self.n_tasks})" if self.name is None else self.name

    @classmethod
    def from_task_pool(cls, task_pool: Array, weights: Array, **kwargs) -> "NoisyLinearRegression":
        assert kwargs["n_tasks"] == task_pool.shape[0]
        task = cls(**kwargs)
        task.task_pool = task_pool
        task.weights = weights
        return task

    def generate_task_pool(self) -> Array:
        key = jax.random.fold_in(self.task_key, 0)
        shape = self.n_tasks, self.n_dims, 1
        tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                              self.distrib_name, self.distrib_param, shape, self.dtype)
        log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                     self.distrib_name, self.distrib_param, self.use_weights, reduce_axis=1)
        #weights = jax.nn.softmax(log_weights, axis=0)
        weights = log_weights
        return tasks, weights

    def generate_data_pool(self) -> Array:
        key = jax.random.fold_in(self.data_key, 0)
        shape = self.n_data, self.n_points, self.n_dims
        data = jax.random.normal(key, shape, self.dtype) * self.data_scale
        return data

    @jax.jit
    def sample_data(self, step: int) -> Array:
        key = jax.random.fold_in(self.data_key, step)
        if self.n_data > 0:
            idxs = jax.random.choice(key, self.n_data, (self.batch_size,))
            data = self.data_pool[idxs]
        else:
            shape = self.batch_size, self.n_points, self.n_dims
            data = jax.random.normal(key, shape, self.dtype) * self.data_scale + self.task_center
        return data

    @jax.jit
    def sample_tasks(self, step: int) -> Array:
        key = jax.random.fold_in(self.task_key, step)
        if self.n_tasks > 0:
            idxs = jax.random.choice(key, self.n_tasks, (self.batch_size,))
            # jax.debug.print("Sampled indices for tasks: {}", idxs)
            tasks = self.task_pool[idxs]
            # log_weights = self.weights[idxs] 
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, self.use_weights, reduce_axis=1)
            weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
        else:
            shape = self.batch_size, self.n_dims, 1
            tasks = sample_distrib(key, self.task_center, self.task_scale, self.clip, 
                                 self.distrib_name, self.distrib_param, shape, self.dtype)
            log_weights = task_log_weights(tasks, self.task_center, self.task_scale, self.clip, 
                                         self.distrib_name, self.distrib_param, self.use_weights, reduce_axis=1)
            weights = jax.nn.softmax(log_weights, axis=0) * self.batch_size  # Scale weights to match batch size
        chex.assert_shape(tasks, (self.batch_size, self.n_dims, 1))
        chex.assert_shape(weights, (self.batch_size, 1))
        # jax.debug.print("Weights sum: {}", jnp.sum(weights))
        # jax.debug.print("Batch statistics: tasks min {}, max {}, mean {}",
        #                jnp.min(tasks), jnp.max(tasks), jnp.mean(tasks))
        return tasks, weights

    @jax.jit
    def evaluate(self, data: Array, tasks: Array, step: int) -> Array:
        targets = (data @ tasks)[:, :, 0]
        key = jax.random.fold_in(self.noise_key, step)
        noise = jax.random.normal(key, targets.shape, self.dtype) * self.noise_scale
        return targets + noise

    @jax.jit
    def generate_attention_mask(self) -> Array:
        """Generate causal attention mask for the sequence with right padding.
        
        Creates a mask of size (2*n_max_points, 2*n_max_points) where:
        - First 2*n_points positions are valid (actual data) 
        - Remaining positions are padded and masked out (right padding)
        - Within valid positions, uses causal attention (can only attend to previous positions)
        """
        effective_seq_len = 2 * self.n_points      # Valid data: positions 0 to this-1
        max_seq_len = 2 * self.n_max_points        # Total padded length
        
        # Start with all positions masked (False)
        mask = jnp.zeros((max_seq_len, max_seq_len), dtype=bool)
        
        # Valid region gets causal attention pattern
        valid_mask = jnp.tril(jnp.ones((effective_seq_len, effective_seq_len))).astype(bool)
        
        # Insert valid causal mask into full mask 
        mask = mask.at[:effective_seq_len, :effective_seq_len].set(valid_mask)
        return mask

    @jax.jit
    def sample_batch(self, step: int) -> tuple[Array, Array, Array, Array]:
        data, (tasks, weights) = self.sample_data(step), self.sample_tasks(step)
        targets = self.evaluate(data, tasks, step)
        attention_mask = self.generate_attention_mask()
        return data, tasks, weights, targets, attention_mask

    @staticmethod
    @jax.jit
    def evaluate_oracle(data: Array, tasks: Array) -> Array:
        targets = (data @ tasks)[:, :, 0]
        return targets

    def get_default_eval_tasks(
            self, batch_size: int, task_seed: int, data_seed: int, noise_seed: int, eval_n_points: List[int], task_centers: List[float] | None = None, **kwargs
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
        config["n_max_points"] = self.n_max_points
        config["use_weights"] = False
        eval_tasks = []
        n_points = eval_n_points
        assert n_points <= self.n_max_points, f"n_points {n_points} exceeds n_max_points {self.n_max_points}"
        config["n_points"] = n_points
        # Test  with fresh tasks from training distribution
        name = f"Test tasks"
        config["name"] = name
        eval_tasks.append(self.__class__(**config))

        # Test with same tasks as training distribution
        if self.n_tasks > 0:
            name = f"Train tasks"
            config["n_tasks"] = self.n_tasks
            config["name"] = name
            eval_tasks.append(NoisyLinearRegression.from_task_pool(**config, task_pool=self.task_pool.copy(), weights=self.weights.copy()))
        
        config["n_tasks"] = 0  # Reset for fresh tasks

        # Test with fixed task centers
        if task_centers is not None:
            for task_center in task_centers:
                config["task_center"] = task_center
                # config["task_scale"] = 0.
                config["clip"] = None
                name = f"Fixed task {task_center}"
                config["name"] = name
                eval_tasks.append(self.__class__(**config))
        return eval_tasks

    def get_default_eval_models(self) -> list[Model]:
        if self.eval_ridge:
            models = [get_model(name="ridge", lam=self.noise_scale**2 / self.task_scale**2, dtype=self.dtype)]
            return models
        else:
            return []

    def _tree_flatten(self):
        # Dynamic values (arrays, keys, and values that can change)
        children = (
            self.data_key,
            self.task_key, 
            self.noise_key,
            self.task_pool,
            self.weights,
            self.data_pool,
            self.data_scale,
            self.task_scale,
            self.noise_scale,
            self.task_center,
            self.clip,
            self.distrib_param,
        )
        
        # Static values (configuration that doesn't change during execution)
        aux_data = {
            'n_tasks': self.n_tasks,
            'n_data': self.n_data,
            'n_dims': self.n_dims,
            'n_points': self.n_points,
            'batch_size': self.batch_size,
            'data_seed': self.data_seed,
            'task_seed': self.task_seed,
            'noise_seed': self.noise_seed,
            'dtype': self.dtype,
            'n_max_points': self.n_max_points,
            'name': self.name,
            'eval_ridge': self.eval_ridge,
            'use_weights': self.use_weights,
            'distrib_name': self.distrib_name,
        }
        
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        (data_key, task_key, noise_key, task_pool, weights, data_pool,
         data_scale, task_scale, noise_scale, task_center, clip, distrib_param) = children
        
        # Create object with aux_data parameters and placeholder scale values
        obj = cls(data_scale=1.0, task_scale=1.0, noise_scale=1.0, 
                 task_center=0.0, clip=None, distrib_param=None, **aux_data)
        
        # Set the dynamic values
        obj.data_key = data_key
        obj.task_key = task_key
        obj.noise_key = noise_key
        obj.task_pool = task_pool
        obj.weights = weights
        obj.data_pool = data_pool
        obj.data_scale = data_scale
        obj.task_scale = task_scale
        obj.noise_scale = noise_scale
        obj.task_center = task_center
        obj.clip = clip
        obj.distrib_param = distrib_param
        
        return obj


# Register NoisyLinearRegression as a PyTree
tree_util.register_pytree_node(NoisyLinearRegression,
                               NoisyLinearRegression._tree_flatten,
                               NoisyLinearRegression._tree_unflatten)

########################################################################################################################
# Get Task                                                                                                             #
########################################################################################################################

Task = NoisyLinearRegression


def get_task(name: str, **kwargs) -> Task:
    tasks = {"noisy_linear_regression": NoisyLinearRegression}
    return tasks[name](**kwargs)
