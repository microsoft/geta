"""Individual components of Transformer model.

This module contains the individual components of the Transformer model.
"""

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization module."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize Root Mean Square Layer Normalization module.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float): A small value to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x) -> torch.Tensor:
        """Calculate the RMSNorm.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> torch.Tensor:
        """Perform forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            The normalized tensor multiplied by the weight parameter.
        """
        output = self._norm(x.float()).type_as(x)

        return output * self.weight


def rotary_mat(
    hidden_size: int,
    n_heads: int,
    max_seq_len: int,
    theta: float = 10000.0,
    head_scale=1.0,
    device=torch.device("cuda"),
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the rotary matrices for the sequence.

    Args:
        hidden_size (int): The size of the hidden layer.

        n_heads (int): The number of attention heads.

        max_seq_len (int): The maximum sequence length.

        theta (float): The value of theta.

        head_scale (float): The scale of the head.

        device (torch.device): The device to use.

        dtype (torch.dtype): The data type to use.

    Returns:
        The cosine and sine matrices.
    """
    head_dim = head_scale * hidden_size / n_heads
    pos = torch.arange(0, 2 * (head_dim // 2), step=2, device=device, dtype=dtype)
    freqs = 1.0 / (theta ** (pos / head_dim))

    idx = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(idx, freqs)

    cos = torch.reshape(torch.cos(freqs), [1, max_seq_len, 1, -1])
    sin = torch.reshape(torch.sin(freqs), [1, max_seq_len, 1, -1])
    dtype = torch.get_default_dtype()

    return cos.to(dtype), sin.to(dtype)


class RotaryEmbedding(torch.nn.Module):
    """Rotary Embedding module."""

    def __init__(self):
        """Initialize RotaryEmbedding."""
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pos: int,
        interleaved: bool = False,
    ):
        """Rotate the tensor.

        Args:
            x (torch.Tensor): The input tensor.

            cos (torch.Tensor): The cosine matrix.

            sin (torch.Tensor): The sine matrix.

            pos (int): The position.

            interleaved (bool): Whether the tensor is interleaved.
        Returns:
            The rotated tensor.
        """
        # Dimension of x is [batch_size, seq_len, n_heads, head_dim]
        rot_dim = 2 * cos.shape[3]

        # Dolly requires partial rotation
        x_rot = x[:, :, :, :rot_dim]

        if interleaved:
            x1 = x_rot[:, :, :, 0::2]
            x2 = x_rot[:, :, :, 1::2]
        else:
            half = x_rot.shape[-1] // 2
            x1 = x[:, :, :, 0:half]
            x2 = x[:, :, :, half : 2 * half]

        seq_len = x.shape[1]
        cos_x = cos[:, pos : pos + seq_len, :, :]
        sin_x = sin[:, pos : pos + seq_len, :, :]

        real = cos_x[..., : x1.shape[-1]] * x1 - sin_x[..., : x2.shape[-1]] * x2
        imag = sin_x[..., : x1.shape[-1]] * x1 + cos_x[..., : x2.shape[-1]] * x2

        if interleaved:
            x_rot[:, :, :, 0::2] = real
            x_rot[:, :, :, 1::2] = imag
        else:
            x_rot = torch.cat((real, imag), dim=-1)

        return torch.cat((x_rot, x[:, :, :, rot_dim:]), dim=-1)


class SelfAttention(nn.Module):
    """Self-Attention module."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        device: torch.device | None = None,
        use_biases: bool = True,
        interleaved: bool = False,
    ) -> None:
        """Initialize Self-Attention module.

        Args:
            hidden_size (int): The size of the hidden layer.

            n_heads (int): The number of attention heads.

            scale_type (str): The type of scaling.

            device (torch.device): The device to use.

            use_biases (bool): Whether to use biases.

            interleaved (bool): Whether the tensor is interleaved.
        """
        super().__init__()
        self.query_w = nn.Linear(
            hidden_size, hidden_size, bias=use_biases, device=device
        )
        self.key_w = nn.Linear(hidden_size, hidden_size, bias=use_biases, device=device)
        self.value_w = nn.Linear(
            hidden_size, hidden_size, bias=use_biases, device=device
        )
        self.attn_w = nn.Linear(
            hidden_size, hidden_size, bias=use_biases, device=device
        )
        self.rotary_emb = RotaryEmbedding()

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = int(hidden_size / n_heads)

        if scale_type == "HeadDim":
            self.scale = self.head_dim
        elif scale_type == "SquareRootHeadDim":
            self.scale = np.sqrt(self.head_dim)
        else:
            raise ValueError(f"Unknown scale type {scale_type}")

        self.interleaved = interleaved

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the Self-Attention module.

        Args:
            x (torch.Tensor): The input tensor.
            attn_mask (torch.Tensor): The attention mask.
            cos (torch.Tensor): The cosine matrix.
            sin (torch.Tensor): The sine matrix.
            k_cache (torch.Tensor): The key cache.
            v_cache (torch.Tensor): The value cache.
            pos (int): The position.
            layer_id (int): The layer ID.
        Returns:
            (torch.Tensor) The attention values.
        """
        # Dimension of x is [batch_size, seq_len, hidden_size]
        # Dimension of attn_mask is [batch_size, max_seq_len, max_seq_len]
        # Dimension of k_cache and v_cache is
        #   [batch_size, n_layers, pos, n_heads, head_dim]
        query = self.query_w(x)
        key = self.key_w(x)
        value = self.value_w(x)

        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Split the attention heads
        query = torch.reshape(query, [batch_size, seq_len, self.n_heads, self.head_dim])
        key = torch.reshape(key, [batch_size, seq_len, self.n_heads, self.head_dim])
        value = torch.reshape(value, [batch_size, seq_len, self.n_heads, self.head_dim])

        # Apply rotary positional embedding
        query = self.rotary_emb(query, cos, sin, pos, self.interleaved)
        key = self.rotary_emb(key, cos, sin, pos, self.interleaved)
        query = query.to(x.dtype)
        key = key.to(x.dtype)

        k_out = key
        v_out = value

        # Append new entries to the end of k, v cache
        pruned_size = key.shape[-2]
        key = torch.cat((k_cache[:, layer_id, :, :pruned_size, :], key), dim=1)
        value = torch.cat((v_cache[:, layer_id, :, :pruned_size, :], value), dim=1)
        # key = torch.cat((k_cache[:, layer_id, :, :, :key.shape[-1]], key), dim=1)
        # value = torch.cat((v_cache[:, layer_id, :, :, :value.shape[-1]], value), dim=1)

        query = query.permute([0, 2, 1, 3]).reshape(
            [batch_size * self.n_heads, seq_len, self.head_dim]
        )
        key = key.permute([0, 2, 3, 1]).reshape(
            [batch_size * self.n_heads, self.head_dim, seq_len + pos]
        )
        value = value.permute([0, 2, 1, 3]).reshape(
            [batch_size * self.n_heads, seq_len + pos, self.head_dim]
        )

        # Calculate attention scores
        score = torch.matmul(query, key) / self.scale

        # Dimension of score is [batch_size * n_heads, seq_len, pos + seq_len]
        batched_mask = torch.cat(
            [
                attn_mask[:, pos : pos + seq_len, : pos + seq_len]
                for _ in range(self.n_heads)
            ]
        )
        score = score + batched_mask

        # Calculate attention values
        prob = f.softmax(score, dim=-1)
        attn = torch.matmul(prob, value)

        # Merge attention heads
        attn = attn.reshape(batch_size, self.n_heads, seq_len, self.head_dim)
        attn = attn.permute([0, 2, 1, 3]).reshape([batch_size, seq_len, -1])

        return self.attn_w(attn), k_out, v_out


class ProjLayer(nn.Module):
    """The projection layer."""

    def __init__(self, hidden_size: int, device: torch.device | None = None) -> None:
        """Create a new instance of ProjLayer.

        Creates a new instance of ProjLayer with the given hidden size and at a
        given device.

        Arguments:
            hidden_size -- Size of projection layer.

        Keyword Arguments:
            device -- _description_ (default: {None})
        """
        super().__init__()
        self.hidden_size = hidden_size

        self.to_4h = nn.Linear(hidden_size, 4 * hidden_size, bias=True, device=device)
        self.to_h = nn.Linear(4 * hidden_size, hidden_size, bias=True, device=device)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward pass of the projection layer.

        Arguments:
            x -- (torch.Tensor) Input tensor.

        Returns:
            (torch.Tensor) Output tensor.
        """
        # Dimension of x is [batch_size, seq_len, hidden_size]
        return self.to_h(self.gelu(self.to_4h(x)))


class ProjLayerSiluMatMul(nn.Module):
    """The projection layer with SiLU activation function."""

    def __init__(
        self,
        in_feature_size: int,
        hidden_feature_size: int,
        device: torch.device | None = None,
    ) -> None:
        """Create a new instance of ProjLayerSiluMatMul."""
        super().__init__()
        self.hidden_feature_size = hidden_feature_size
        self.in_feature_size = in_feature_size

        self.w1 = nn.Linear(
            in_feature_size, hidden_feature_size, bias=False, device=device
        )
        self.w2 = nn.Linear(
            hidden_feature_size, in_feature_size, bias=False, device=device
        )
        self.w3 = nn.Linear(
            in_feature_size, hidden_feature_size, bias=False, device=device
        )

    def forward(self, x):
        """Apply forward pass of the projection layer.

        Arguments:
            x -- (torch.Tensor) Input tensor.

        Returns:
            (torch.Tensor) Output tensor.
        """
        w1x = self.w1(x)

        return self.w2(w1x * f.sigmoid(w1x) * self.w3(x))
