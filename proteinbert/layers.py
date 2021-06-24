import jax.numpy as jnp
from chex import Array
from einops import rearrange
from einops.einops import repeat
from flax import linen as nn
from flax.linen.activation import tanh


class GlobalLinearSelfAttention(nn.Module):

    dim: int
    dim_head: int
    heads: int

    @nn.compact
    def __call__(self, feats) -> Array:

        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5
        h = self.heads

        qkv = nn.Dense(features=3 * h * self.dim_head, use_bias=False)(feats)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = nn.softmax(q, dim=-1)
        k = nn.softmax(k, dim=-2)
        q = q * scale

        context = jnp.einsum("b h n d, b h n e -> b h d e", k, v)
        out = jnp.einsum("b h d e, b h n d -> b h n e", context, q)
        out = rearrange(out, "b h n d -> b n (h d)")
        return nn.Dense(featurers=inner_dim, use_bias=False)(out)


class CrossAttention(nn.Module):

    dim: int
    dim_keys: int
    dim_out: int
    heads: int
    dim_head: int = (64,)
    qk_activation = tanh()

    @nn.compact
    def __call__(self, x, context) -> Array:

        b, h = x.shape[0], self.heads
        self.scale = self.dim_head ** -0.5
        inner_dim = self.dim_head * self.heads

        q = nn.Dense(features=inner_dim, use_bias=False)(x)
        kv = nn.Dense(features=inner_dim * 2, use_bias=False)(context)
        k, v = jnp.split(kv, 2, axis=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        null_k, null_v = map(
            lambda t: repeat(t, "d -> b h () d", b=b, h=h),
            (self.null_key, self.null_value),
        )

        k = jnp.concatenate((null_k, k), dim=-2)
        v = jnp.concatenate((null_v, v), dim=-2)
        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = jnp.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = nn.softmax(sim, dim=-1)
        out = jnp.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return nn.Dense(features=self.dim_out)(out)
