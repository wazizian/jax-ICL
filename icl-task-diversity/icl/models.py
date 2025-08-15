from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import Array
import chex

import icl.utils as u
from icl.gpt2 import GPT2Config, GPT2Model, init_fn


########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################


def get_model_name(model):
    if isinstance(model, Ridge):
        return "Ridge"
    elif isinstance(model, DiscreteMMSE):
        return "dMMSE"
    elif isinstance(model, Transformer):
        return "Transformer"
    elif isinstance(model, SingleSeqTransformer):
        return "SingleSeqTransformer"
    elif isinstance(model, LastValue):
        return "LastValue"
    else:
        raise ValueError(f"model type={type(model)} not supported")


########################################################################################################################
# Transformer                                                                                                          #
########################################################################################################################


class Transformer(nn.Module):
    n_points: int
    n_layer: int
    n_embd: int
    n_head: int
    seed: int
    dtype: Any
    use_ln: bool = True
    use_linear_attention: bool = False

    def setup(self):
        config = GPT2Config(
            block_size=2 * self.n_points,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dtype=self.dtype,
            use_ln=self.use_ln,
            use_linear_attention=self.use_linear_attention,
        )
        self._in = nn.Dense(self.n_embd, False, self.dtype, kernel_init=init_fn)
        self._h = GPT2Model(config)
        self._out = nn.Dense(1, False, self.dtype, kernel_init=init_fn)

    def __call__(self, data: Array, targets: Array, attention_mask: Array, training: bool = False) -> Array:
        # Batch size
        batch_size = data.shape[0]
        # Get actual sequence length before padding
        actual_seq_len = data.shape[1]
        # Data features
        n_features = data.shape[2]
        
        chex.assert_shape(data, (batch_size, actual_seq_len, n_features))
        chex.assert_shape(targets, (batch_size, actual_seq_len))
        
        # Pad input sequence to match the model's expected block_size
        target_seq_len = self.n_points  # Expected number of data points

        chex.assert_shape(attention_mask, (batch_size, 2 * self.n_points, 2 * self.n_points))

        input_seq = u.to_seq(data, targets, target_seq_len=target_seq_len)

        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, attention_mask=attention_mask, training=training)
        preds = self._out(outputs)
        preds = u.seq_to_targets(preds, actual_seq_len=actual_seq_len)
        return preds

########################################################################################################################
# Single Seq Transformer                                                                                                          #
########################################################################################################################


class SingleSeqTransformer(nn.Module):
    n_points: int
    n_layer: int
    n_embd: int
    n_head: int
    seed: int
    dtype: Any
    use_ln: bool = True
    use_linear_attention: bool = False
    n_out: int = 1

    def setup(self):
        config = GPT2Config(
            block_size=self.n_points,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dtype=self.dtype,
            use_ln=self.use_ln,
            use_linear_attention=self.use_linear_attention,
        )
        self._in = nn.Dense(self.n_embd, False, self.dtype, kernel_init=init_fn)
        self._h = GPT2Model(config)
        self._out = nn.Dense(self.n_out, False, self.dtype, kernel_init=init_fn)

    def __call__(self, data: Array, targets: Array, attention_mask: Array, training: bool = False) -> Array:
        # Batch size
        batch_size = data.shape[0]
        # Get actual sequence length before padding
        actual_seq_len = data.shape[1]
        # Target features
        n_data_features = data.shape[2]
        
        chex.assert_shape(data, (batch_size, actual_seq_len, n_data_features))
        
        # Pad input sequence to match the model's expected block_size
        data_seq_len = self.n_points  # Expected number of data points

        chex.assert_shape(attention_mask, (batch_size, self.n_points, self.n_points))

        input_seq = u.pad_sequence(data, target_seq_len=data_seq_len)
        chex.assert_shape(input_seq, (batch_size, self.n_points, n_data_features))

        embds = self._in(input_seq)
        outputs = self._h(input_embds=embds, attention_mask=attention_mask, training=training)
        preds = self._out(outputs)
        chex.assert_shape(preds, (batch_size, self.n_points, n_data_features))

        preds = u.unpad_sequence(preds, actual_seq_len=actual_seq_len)
        chex.assert_shape(preds, (batch_size, actual_seq_len, n_data_features))

        return preds

class LastValue(nn.Module):
    """
    A simple model that returns the last value of the input sequence.
    This is useful for tasks where the last value is the target.
    """

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points x n_dims (float)
        Return:
            batch_size x n_points x n_dims (float)
        """
        batch_size, n_points, n_dims = targets.shape

        init = jnp.zeros((batch_size, 1, n_dims), dtype=targets.dtype)
        preds = jnp.concatenate([init, targets[:, :-1, :]], axis=1)  # batch_size x n_points x n_dims
        chex.assert_shape(preds, (batch_size, n_points, n_dims))

        return preds


########################################################################################################################
# Ridge                                                                                                                #
########################################################################################################################


class Ridge(nn.Module):
    lam: float
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            xs: batch_size x n_points x n_dims (float)
            ys: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        batch_size, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        preds = [jnp.zeros(batch_size, dtype=self.dtype)]
        preds.extend(
            [self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], self.lam) for _i in range(1, n_points)]
        )
        preds = jnp.stack(preds, axis=1)
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, lam: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            lam: (float)
        Return:
            batch_size (float)
        """
        _, _, n_dims = X.shape
        XT = X.transpose((0, 2, 1))  # batch_size x n_dims x i
        XT_Y = XT @ Y  # batch_size x n_dims x 1, @ should be ok (batched matrix-vector product)
        ridge_matrix = jnp.matmul(XT, X, precision=jax.lax.Precision.HIGHEST) + lam * jnp.eye(n_dims, dtype=self.dtype)  # batch_size x n_dims x n_dims
        # batch_size x n_dims x 1
        ws = jnp.linalg.solve(ridge_matrix.astype(jnp.float32), XT_Y.astype(jnp.float32)).astype(self.dtype)
        pred = test_x @ ws  # @ should be ok (batched row times column)
        return pred[:, 0, 0]


########################################################################################################################
# MMSE                                                                                                                #
########################################################################################################################


class DiscreteMMSE(nn.Module):
    scale: float
    task_pool: Array  # n_tasks x n_dims x 1
    dtype: Any

    def __call__(self, data: Array, targets: Array) -> Array:
        """
        Args:
            data: batch_size x n_points x n_dims (float)
            targets: batch_size x n_points (float)
        Return:
            batch_size x n_points (float)
        """
        _, n_points, _ = data.shape
        targets = jnp.expand_dims(targets, -1)  # batch_size x n_points x 1
        W = self.task_pool.squeeze().T  # n_dims x n_tasks  (maybe do squeeze and transpose in setup?)
        preds = [data[:, 0] @ W.mean(axis=1)]  # batch_size
        preds.extend(
            [
                self.predict(data[:, :_i], targets[:, :_i], data[:, _i : _i + 1], W, self.scale)
                for _i in range(1, n_points)
            ]
        )
        preds = jnp.stack(preds, axis=1)  # batch_size x n_points
        return preds

    def predict(self, X: Array, Y: Array, test_x: Array, W: Array, scale: float) -> Array:
        """
        Args:
            X: batch_size x i x n_dims (float)
            Y: batch_size x i x 1 (float)
            test_x: batch_size x 1 x n_dims (float)
            W: n_dims x n_tasks (float)
            scale: (float)
        Return:
            batch_size (float)
        """
        # X @ W is batch_size x i x n_tasks, Y is batch_size x i x 1, so broadcasts to alpha being batch_size x n_tasks
        # alpha = tfd.Normal(0, scale).log_prob(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST)).astype(self.dtype).sum(axis=1)
        alpha = jax.scipy.stats.norm.logpdf(Y - jnp.matmul(X, W, precision=jax.lax.Precision.HIGHEST), loc=0, scale=scale).astype(self.dtype).sum(axis=1)
        # softmax is batch_size x n_tasks, W.T is n_tasks x n_dims, so w_mmse is batch_size x n_dims x 1
        w_mmse = jnp.expand_dims(jnp.matmul(jax.nn.softmax(alpha, axis=1), W.T, precision=jax.lax.Precision.HIGHEST), -1)
        # test_x is batch_size x 1 x n_dims, so pred is batch_size x 1 x 1. NOTE: @ should be ok (batched row times column)
        pred = test_x @ w_mmse
        return pred[:, 0, 0]


########################################################################################################################
# Get Model                                                                                                            #
########################################################################################################################

Model = Transformer | Ridge | DiscreteMMSE


def get_model(name: str, **kwargs) -> Model:
    models = {"transformer": Transformer, "ridge": Ridge, "discrete_mmse": DiscreteMMSE, "single_seq_transformer": SingleSeqTransformer, "last_value": LastValue}
    return models[name](**kwargs)
