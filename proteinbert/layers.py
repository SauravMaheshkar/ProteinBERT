from typing import Any

import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

ATTN_MASK_VALUE = -1e10

Dtype = Any


class GlobalLinearSelfAttention(nn.Module):

    dim: int
    dim_head: int
    heads: int

    @nn.compact
    def __call__(self, feats):

        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5
        h = self.heads

        qkv = nn.Dense(features = 3 * h * self.dim_head, use_bias=False)(feats)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = nn.softmax(q, dim=-1)
        k = nn.softmax(k, dim=-2)
        q = q * scale

        context = jnp.einsum("b h n d, b h n e -> b h d e", k, v)
        out = jnp.einsum("b h d e, b h n d -> b h n e", context, q)
        out = rearrange(out, "b h n d -> b n (h d)")
        return nn.Dense(featurers = inner_dim, use_bias = False)(out)


class CrossAttention(nn.Module):
    def __init__(self) -> None:
        pass
