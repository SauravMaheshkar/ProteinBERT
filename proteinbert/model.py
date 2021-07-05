from typing import Any, Sequence

import jax.numpy as jnp
from chex import Array
from einops import rearrange
from flax import linen as nn

from .layers import Layer
from .utils import Reduce, Sequential

Dtype = Any

__all__ = ["ProteinBERT"]


class ProteinBERT(nn.Module):
    """A Flax Module for the ProteinBERT architecture

    Attributes:
        num_tokens: No of Tokens
        num_annotation: No of Annotations, used as output dimension to get annotation logits
        dim: No of output dimensions for the GlSA Block
        dim_global: No of output dimensions for the Cross Attention Block
        depth: No of layers 
        narrow_conv_kernel: kernel size for narrow conv layer
        wide_conv_kernel: kernel size for wide conv layer
        wide_conv_dilation: kernel dilation
        attn_heads: number of attention heads
        attn_dim_head: dimensionality for the attention heads
        local_to_global_attn: (bool) whether to use Local to Global Attention
        local_self_attn: (bool) whether to use Local Self Attention
        num_global_tokens: No of global tokens
        glu_conv: (bool) whether to use glu
        dtype: the dtype of the computation (default: float32)
    """

    num_tokens: int = 26
    num_annotation: int = 8943
    dim: int = 512
    dim_global: int = 256
    depth: int = 6
    narrow_conv_kernel: int = 9
    wide_conv_kernel: int = 9
    wide_conv_dilation: int = 5
    attn_heads: int = 8
    attn_dim_head: int = 64
    local_to_global_attn: bool = False
    local_self_attn: bool = False
    num_global_tokens: int = 1
    glu_conv: bool = False
    dtype: Dtype = jnp.float32

    def setup(self):
        self.token_emb = nn.Embed(
            num_embeddings=self.num_tokens, features=self.dim, dtype=self.dtype
        )

        self.to_global_emb = nn.Dense(
            features=self.num_global_tokens * self.dim_global, dtype=self.dtype
        )

        self.layers = [
            Layer(
                dim=self.dim,
                dim_global=self.dim_global,
                narrow_conv_kernel=self.narrow_conv_kernel,
                wide_conv_dilation=self.wide_conv_dilation,
                wide_conv_kernel=self.wide_conv_kernel,
                local_to_global_attn=self.local_to_global_attn,
                local_self_attn=self.local_self_attn,
                glu_conv=self.glu_conv,
                dtype=self.dtype,
            )
            for layer in range(self.depth)
        ]

        self.to_token_logits = nn.Dense(features=self.num_tokens, dtype=self.dtype)

        self.to_annotation_logits = Sequential(
            [
                Reduce(pattern="b n d -> b d", reduction="mean"),
                nn.Dense(features=self.num_annotation, dtype=self.dtype),
            ]
        )

    def __call__(self, seq, annotation) -> Sequence[Array]:
        tokens = self.token_emb(seq)

        annotation = self.to_global_emb(annotation)
        annotation = rearrange(annotation, "b (n d) -> b n d", n=self.num_global_tokens)

        for layer in self.layers:
            tokens, annotation = layer(tokens, annotation)

        tokens = self.to_token_logits(tokens)
        annotation = self.to_annotation_logits(annotation)
        return tokens, annotation
