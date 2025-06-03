import flax.linen as nn
import jax.numpy as jnp

class Encoder(nn.Module):
    hidden_dims: tuple = (256, 256)
    proj_hidden: int = 128
    embed_dim: int = 128
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training=True):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        raw = nn.Dense(self.embed_dim)(x)
        z = nn.Dense(self.proj_hidden)(raw)
        z = nn.relu(z)
        if training:
            z = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(z)
        z = nn.Dense(self.embed_dim)(z)
        z = z / (jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        return raw, z 