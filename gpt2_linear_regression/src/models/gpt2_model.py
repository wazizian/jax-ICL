import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
from typing import Optional, Dict, Any
from omegaconf import DictConfig
import math


class GPT2Attention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    config: Dict[str, Any]
    
    def setup(self):
        self.n_head = self.config['n_head']
        self.n_embd = self.config['n_embd']
        self.head_dim = self.n_embd // self.n_head
        
        # Combined projection for q, k, v
        self.c_attn = nn.Dense(
            3 * self.n_embd,
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        
        # Output projection
        self.c_proj = nn.Dense(
            self.n_embd,
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        
        self.attn_dropout = nn.Dropout(rate=self.config['attn_pdrop'])
        self.resid_dropout = nn.Dropout(rate=self.config['resid_pdrop'])
        
    def __call__(self, x, attention_mask=None, deterministic=True):
        batch_size, seq_len, _ = x.shape
        
        # Apply combined linear transformation
        qkv = self.c_attn(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_head, self.head_dim)
        
        # Transpose to (batch_size, n_head, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        attn_weights = jnp.where(causal_mask, attn_weights, -jnp.inf)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attn_weights = jnp.where(attention_mask, attn_weights, -jnp.inf)
        
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Transpose back and reshape
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.n_embd)
        
        # Output projection
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)
        
        return attn_output


class GPT2MLP(nn.Module):
    """Feed-forward network."""
    
    config: Dict[str, Any]
    
    def setup(self):
        self.c_fc = nn.Dense(
            self.config['n_inner'],
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        self.c_proj = nn.Dense(
            self.config['n_embd'],
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        self.dropout = nn.Dropout(rate=self.config['resid_pdrop'])
        
    def __call__(self, x, deterministic=True):
        x = self.c_fc(x)
        x = nn.gelu(x, approximate=True)  # GELU activation
        x = self.c_proj(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class GPT2Block(nn.Module):
    """Transformer block."""
    
    config: Dict[str, Any]
    
    def setup(self):
        self.ln_1 = nn.LayerNorm(epsilon=self.config['layer_norm_epsilon'])
        self.attn = GPT2Attention(self.config)
        self.ln_2 = nn.LayerNorm(epsilon=self.config['layer_norm_epsilon'])
        self.mlp = GPT2MLP(self.config)
        
    def __call__(self, x, attention_mask=None, deterministic=True):
        # Pre-norm architecture
        attn_output = self.attn(
            self.ln_1(x), 
            attention_mask=attention_mask, 
            deterministic=deterministic
        )
        x = x + attn_output
        
        mlp_output = self.mlp(self.ln_2(x), deterministic=deterministic)
        x = x + mlp_output
        
        return x




class GPT2LinearModel(nn.Module):
    """GPT2 model without vocabulary - uses linear read-in and read-out layers."""
    
    config: Dict[str, Any]
    
    def setup(self):
        # Read-in layer: input_dim -> embed_dim
        self.read_in = nn.Dense(
            self.config['n_embd'],
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        
        # Position embeddings (keep these for positional encoding)
        self.wpe = nn.Embed(
            self.config['n_positions'],
            self.config['n_embd'],
            embedding_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        
        self.drop = nn.Dropout(rate=self.config['embd_pdrop'])
        
        # Transformer blocks
        self.h = [GPT2Block(self.config) for _ in range(self.config['n_layer'])]
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(epsilon=self.config['layer_norm_epsilon'])
        
        # Read-out layer: embed_dim -> output_dim
        self.read_out = nn.Dense(
            self.config.get('output_dim', self.config['n_embd']),
            use_bias=True,
            kernel_init=initializers.normal(stddev=self.config['initializer_range'])
        )
        
    def __call__(self, inputs, attention_mask=None, deterministic=True):
        batch_size, seq_len, input_dim = inputs.shape
        
        # Read-in: map input to embedding space
        hidden_states = self.read_in(inputs)
        
        # Create position ids
        position_ids = jnp.arange(seq_len)[None, :]
        position_ids = jnp.broadcast_to(position_ids, (batch_size, seq_len))
        
        # Add position embeddings
        position_embeddings = self.wpe(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.drop(hidden_states, deterministic=deterministic)
        
        # Pass through transformer blocks
        for block in self.h:
            hidden_states = block(
                hidden_states, 
                attention_mask=attention_mask, 
                deterministic=deterministic
            )
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Read-out: map embedding to output space
        outputs = self.read_out(hidden_states)
        
        return outputs


def create_model(config: DictConfig) -> GPT2LinearModel:
    """Factory function to create GPT2 linear model."""
    model_config = dict(config.model.config)
    return GPT2LinearModel(config=model_config)


