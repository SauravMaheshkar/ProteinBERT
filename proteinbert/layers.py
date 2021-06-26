from typing import Any, Sequence

import jax.numpy as jnp
from attention import CrossAttention, GlobalLinearSelfAttention
from chex import Array
from einops import rearrange
from flax import linen as nn
from utils import Rearrange, Reduce, Residual, Sequential

Dtype = Any

__all__ = ["Layer"]


class Layer(nn.Module):
    """A Flax Module to act as a layer in ProteinBERT

    Attributes:
        dim: No of output dimensions for the GlSA Block
        dim_global: No of output dimensions for the Cross Attention Block
        narrow_conv_kernel: kernel size for narrow conv layer
        wide_conv_kernel: kernel size for wide conv layer
        wide_conv_dilation: kernel dilation
        attn_heads: number of attention heads
        attn_dim_head: dimensionality for the attention heads
        attn_qk_activation: Activation function for Querys and Keys in the Cross Attention Module
        local_to_global_attn: (bool) whether to use Local to Global Attention
        local_self_attn: (bool) whether to use Local Self Attention
        glu_conv: (bool) whether to use glu
        dtype: the dtype of the computation (default: float32)
    """

    dim: int
    dim_global: int
    narrow_conv_kernel: int = 9
    wide_conv_kernel: int = 9
    wide_conv_dilation: int = 5
    attn_heads: int = 8
    attn_dim_head: int = 64
    attn_qk_activation = nn.activation.tanh()
    local_to_global_attn: bool = False
    local_self_attn: bool = False
    glu_conv: bool = False
    dtype: Dtype = jnp.float32

    def setup(self):
        self.seq_self_attn = (
            GlobalLinearSelfAttention(
                dim=self.dim,
                dim_head=self.attn_dim_head,
                heads=self.attn_heads,
                dtype=self.dtype,
            )
            if self.local_self_attn
            else None
        )

        self.narrow_conv = Sequential(
            nn.Conv(
                featurers=self.dim,
                kernel_size=self.narrow_conv_kernel,
                padding=self.narrow_conv_kernel // 2,
                dtype=self.dtype,
            ),
            nn.gelu() if not self.glu_conv else nn.glu(axis=1),
        )

        wide_conv_padding = (
            self.wide_conv_kernel
            + (self.wide_conv_kernel - 1) * (self.wide_conv_dilation - 1)
        ) // 2

        self.wide_conv = Sequential(
            nn.Conv(
                features=self.dim,
                kernel_size=self.wide_conv_kernel,
                kernel_dilation=self.wide_conv_dilation,
                padding=wide_conv_padding,
                dtype=self.dtype,
            ),
            nn.gelu() if not self.glu_conv else nn.glu(axis=1),
        )

        if self.local_to_global_attn:
            self.extract_global_info = CrossAttention(
                dim=self.dim,
                dim_keys=self.dim_global,
                dim_out=self.dim,
                heads=self.attn_heads,
                dim_head=self.attn_dim_head,
                dtype=self.dtype,
            )
        else:
            self.extract_global_info = Sequential(
                Reduce("b n d -> b d", "mean"),
                nn.Dense(self.dim, dtype=self.dtype),
                nn.gelu(),
                Rearrange("b d -> b () d"),
            )

        self.local_norm = nn.LayerNorm()

        self.local_feedforward = Sequential(
            Residual(Sequential(nn.Dense(self.dim, dtype=self.dtype), nn.gelu(),)),
            nn.LayerNorm(dtype=self.dtype),
        )

        self.global_attend_local = CrossAttention(
            dim=self.dim_global,
            dim_out=self.dim_global,
            dim_keys=self.dim,
            heads=self.attn_heads,
            dim_head=self.attn_dim_head,
            qk_activation=self.attn_qk_activation,
        )

        self.global_dense = Sequential(
            nn.Dense(self.dim_global, dtype=self.dtype), nn.gelu()
        )

        self.global_norm = nn.LayerNorm()

        self.global_feedforward = Sequential(
            Residual(
                Sequential(nn.Dense(self.dim_global, dtype=self.dtype), nn.gelu())
            ),
            nn.LayerNorm(dtype=self.dtype),
        )

    @nn.compact
    def __call__(self, tokens, annotation) -> Sequence[Array]:
        if self.local_to_global_attn:
            global_info = self.extract_global_info(tokens, annotation)
        else:
            global_info = self.extract_global_info(annotation)

        # Process Protein Sequences
        global_linear_attn = (
            self.seq_self_attn(tokens) if self.seq_self_attn is not None else 0
        )

        conv_input = rearrange(tokens, "b n d -> b d n")

        narrow_out = self.narrow_conv(conv_input)
        narrow_out = rearrange(narrow_out, "b d n -> b n d")
        wide_out = self.wide_conv(conv_input)
        wide_out = rearrange(wide_out, "b d n -> b n d")

        tokens = tokens + narrow_out + wide_out + global_info + global_linear_attn
        tokens = self.local_norm(tokens)

        tokens = self.local_feedforward(tokens)

        # Process Annotations
        local_info = self.global_attend_local(annotation, tokens)
        annotation = self.global_dense(annotation)
        annotation = self.global_norm(annotation)
        annotation = self.global_feedforward(annotation)

        return tokens, annotation
