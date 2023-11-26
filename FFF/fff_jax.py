import jax
import jax.numpy as jnp
from jax import random
from math import floor, log2, sqrt
from flax import linen as nn
from typing import Optional

INIT_STRAT = 'hyperspherical-shell'

class FFF(nn.Module):
    nIn: int
    nOut: int
    depth: Optional[int] = None

    def setup(self):
        self.n_nodes = 2 ** (self.depth or int(floor(log2(self.nIn)))) - 1

        if INIT_STRAT == 'gaussian':
            depth = self.depth or int(floor(log2(self.nIn)))
            init_factor_I1 = 1 / sqrt(self.nIn)
            init_factor_I2 = 1 / sqrt(depth + 1)

            key1, key2 = random.split(random.PRNGKey(0))
            self.w1s = self.param('w1s', lambda rng, shape: random.normal(rng, shape) * init_factor_I1, (self.n_nodes, self.nIn))
            self.w2s = self.param('w2s', lambda rng, shape: random.normal(rng, shape) * init_factor_I2, (self.n_nodes, self.nOut))

        if INIT_STRAT == 'hyperspherical-shell':
            key1, key2 = random.split(random.PRNGKey(0), 2)
            self.w1s = self.param('w1s', lambda rng, shape: random.normal(rng, shape), (self.n_nodes, self.nIn))
            self.w1s = nn.normalize(self.w1s, axis=-1)
            self.w2s = self.param('w2s', lambda rng, shape: random.normal(rng, shape), (self.n_nodes, self.nOut))
            self.w2s = nn.normalize(self.w2s, axis=-1)

    def __call__(self, x):
        batch_size = x.shape[0]
        current_node = jnp.zeros(batch_size, dtype=jnp.int32)
        y = jnp.zeros((batch_size, self.nOut))
        depth = self.depth or int(floor(log2(self.nIn)))

        for i in range(depth):
            lambda_ = jnp.einsum("bi,bi->b", x, self.w1s[current_node])
            y += jnp.einsum("b,bj->bj", lambda_, self.w2s[current_node])
            plane_choice = (lambda_ > 0).astype(jnp.int32)
            current_node = (current_node * 2) + 1 + plane_choice

        return y
