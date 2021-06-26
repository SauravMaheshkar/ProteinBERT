from typing import Any

import jax.numpy as jnp
from chex import Array
from einops import rearrange, repeat
from flax import linen as nn

__all__ = ["GlobalLinearSelfAttention", "CrossAttention"]

Dtype = Any


class GlobalLinearSelfAttention(nn.Module):
    """A Global Linear Self Attention Layer

    Attributes:
        dim: No of output dimensions for this Block
        dim_head: No of dimensions for each head
        heads: number of heads
        dtype: the dtype of the computation (default: float32)
        kernel_init: initializer function for the weight matrix (default: lecun normal)
        bias_init: initializer function for the bias (default: zeros)
    """

    dim: int
    dim_head: int
    heads: int
    dtype: Dtype = jnp.float32
    kernel_init = nn.initializers.lecun_normal()
    bias_init = nn.initializers.zeros

    @nn.compact
    def __call__(self, feats) -> Array:

        scale = self.dim_head ** -0.5
        h = self.heads

        qkv = nn.Dense(
            features=3 * h * self.dim_head,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="GLSA QKV",
        )(feats)

        q, k, v = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        q = nn.softmax(q, axis=-1)
        k = nn.softmax(k, axis=-2)
        q = q * scale

        context = jnp.einsum("b h n d, b h n e -> b h d e", k, v)
        out = jnp.einsum("b h d e, b h n d -> b h n e", context, q)
        out = rearrange(out, "b h n d -> b n (h d)")

        return nn.Dense(
            featurers=self.dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="GLSA Output",
        )(out)


class CrossAttention(nn.Module):
    """A Cross Attention Layer

    Attributes:
        dim_out: No of output dimensions for this Block
        heads: number of heads
        dim_head: No of dimensions for each head
        qk_activation: activation function for Query and Keys (default: tanh)
        dtype: the dtype of the computation (default: float32)
        kernel_init: initializer function for the weight matrix (default: lecun normal)
        bias_init: initializer function for the bias (default: zeros)
    """

    dim_out: int
    heads: int
    dim_head: int = 64
    qk_activation = nn.activation.tanh()
    dtype: Dtype = jnp.float32
    kernel_init = nn.initializers.lecun_normal()
    bias_init = nn.initializers.zeros

    @nn.compact
    def __call__(self, x, context) -> Array:

        b, h = x.shape[0], self.heads
        self.scale = self.dim_head ** -0.5
        inner_dim = self.dim_head * self.heads

        q = nn.Dense(
            features=inner_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="CA Q",
        )(x)

        kv = nn.Dense(
            features=inner_dim * 2,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="CA KV",
        )(context)

        k, v = jnp.split(kv, 2, axis=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        null_k, null_v = map(
            lambda t: repeat(t, "d -> b h () d", b=b, h=h),
            (self.null_key, self.null_value),
        )

        k = jnp.concatenate((null_k, k), axis=-2)
        v = jnp.concatenate((null_v, v), axis=-2)
        q, k = map(lambda t: self.qk_activation(t), (q, k))

        sim = jnp.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return nn.Dense(
            features=self.dim_out,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            name="CA Output",
        )(out)
