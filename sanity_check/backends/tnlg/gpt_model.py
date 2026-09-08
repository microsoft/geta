"""Compontents for GPT-style language models.

This module defines the TransformerLayer, Transformer, and LanguageModel classes
for a GPT-style language model.

TransformerLayer implements a single layer of the Transformer model, consisting
of a self-attention mechanism and a feedforward neural network.

Transformer implements the entire Transformer model, consisting of multiple
TransformerLayer blocks.

LanguageModel is a wrapper class that adds an embedding layer to the Transformer
model, and is used for language modeling tasks.

The classes use various components defined in the gpt_components module,
including SelfAttention, ProjLayer, rotary_mat, ProjLayerSiluMatMul, and
RMSNorm.

The classes are implemented as PyTorch nn.Module subclasses, and can be used for
training and inference on language modeling tasks.
"""

import torch
from torch import nn

from .gpt_components import (
    ProjLayer,
    ProjLayerSiluMatMul,
    RMSNorm,
    SelfAttention,
    rotary_mat,
)


class TransformerLayer(nn.Module):
    """
    Single layer of the Transformer model.

    A single layer of the Transformer model, consisting of a self-attention
    mechanism and a feedforward neural network.

    Attributes:
        attn_norm (nn.Module): The normalization layer applied to the attention
        output.

        proj_norm (nn.Module): The normalization layer applied to the
        feedforward output.

        attention (SelfAttention): The self-attention mechanism.

        proj (nn.Module): The feedforward neural network.

        model_type (str): The type of model used.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        scale_type: str,
        device: torch.device | None = None,
        use_biases: bool = True,
        interleaved: bool = False,
        model_type: str = "TNLG",
        use_silu_matmul: bool = False,
    ) -> None:
        """Initialize the TransformerLayer.

        Args:
            hidden_size (int): The number of hidden units in the layer.

            n_heads (int): The number of attention heads.

            scale_type (str): The type of scaling to use for the attention
            scores.

            device (torch.device, optional): The device to use for the layer's
            computations. Defaults to None.

            use_biases (bool, optional): Whether to use biases in the attention
            mechanism. Defaults to True.

            interleaved (bool, optional): Whether to use interleaved
            self-attention. Defaults to False.

            model_type (str, optional): The type of model to use. Must be one of
            "TNLG", "Llama", or "Dolly". Defaults to "TNLG".

            use_silu_matmul (bool, optional): Whether to use a SiLU activation
            function in the feedforward network. Defaults to False.

        """
        super().__init__()
        if use_silu_matmul:
            # these should have variable eps.
            self.attn_norm = RMSNorm(hidden_size, eps=1e-5)
            self.proj_norm = RMSNorm(hidden_size, eps=1e-5)
        else:
            self.attn_norm = nn.LayerNorm(hidden_size, device=device)
            self.proj_norm = nn.LayerNorm(hidden_size, device=device)

        self.attention = SelfAttention(
            hidden_size,
            n_heads,
            scale_type,
            device=device,
            use_biases=use_biases,
            interleaved=interleaved,
        )
        if use_silu_matmul:
            proj_dim = hidden_size * 4
            proj_dim = int(2 * proj_dim / 3)
            proj_dim = 256 * ((proj_dim + 256 - 1) // 256)
            self.proj = ProjLayerSiluMatMul(hidden_size, proj_dim, device=device)
        else:
            self.proj = ProjLayer(hidden_size, device=device)

        self.model_type = model_type

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
        """
        Forward pass of the GPTModel.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len,
            hidden_size].

            attn_mask (torch.Tensor): Attention mask tensor of shape
            [batch_size, n_heads, seq_len, seq_len].

            cos (torch.Tensor): Cosine tensor of shape [seq_len, n_heads,
            head_dim].

            sin (torch.Tensor): Sine tensor of shape [seq_len, n_heads,
            head_dim].

            k_cache (torch.Tensor): Key cache tensor of shape [batch_size,
            n_layers, pos, n_heads, head_dim].

            v_cache (torch.Tensor): Value cache tensor of shape [batch_size,
            n_layers, pos, n_heads, head_dim].

            pos (int): Position of the current layer.

            layer_id (int): ID of the current layer.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the output tensor of shape [batch_size, seq_len, hidden_size], the
            key cache tensor of shape [batch_size, n_layers, pos+1, n_heads,
            head_dim], and the value cache tensor of shape [batch_size,
            n_layers, pos+1, n_heads, head_dim].
        """
        # Dimension of x is [batch_size, seq_len, hidden_size] Dimension of
        # k_cache and v_cache is [batch_size, n_layers, pos, n_heads, head_dim]
        h, k_out, v_out = self.attention(
            self.attn_norm(x), attn_mask, cos, sin, k_cache, v_cache, pos, layer_id
        )

        if self.model_type == "TNLG" or self.model_type == "Llama":
            h = x + h
            return h + self.proj(self.proj_norm(h)), k_out, v_out
        elif self.model_type == "Dolly":
            return x + h + self.proj(self.proj_norm(x)), k_out, v_out
        else:
            raise ValueError("model_type must be either TNLG, Llama, or Dolly")


class Transformer(nn.Module):
    """A transformer model that consists of multiple transformer layers."""

    def __init__(
        self,
        n_layers: int,
        hidden_size: int,
        n_heads: int,
        max_seq_len: int,
        scale_type: str,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        use_biases: bool = True,
        interleaved: bool = False,
        model_type: str = "TNLG",
        head_scale: float = 1.0,
        use_silu_matmul: bool = False,
    ) -> None:
        """
        Initialize the Transformer.

        Args:
            n_layers (int): The number of transformer layers.

            hidden_size (int): The hidden size of the transformer.

            n_heads (int): The number of attention heads.

            max_seq_len (int): The maximum sequence length.

            scale_type (str): The type of scaling to use for the attention
            scores.

            device (torch.device, optional): The device to use for the model.
            Defaults to torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu").

            use_biases (bool, optional): Whether to use biases in the linear
            layers. Defaults to True.

            interleaved (bool, optional): Whether to use interleaved linear
            layers. Defaults to False.

            model_type (str, optional): The type of the model. Must be either
            "TNLG", "Llama", or "Dolly". Defaults to "TNLG".

            head_scale (float, optional): The scale factor for the attention
            heads. Defaults to 1.0.

            use_silu_matmul (bool, optional): Whether to use SiLU activation
            function in the linear layers. Defaults to False.
        """
        super().__init__()

        cos, sin = rotary_mat(
            hidden_size, n_heads, max_seq_len, head_scale=head_scale, device=device
        )
        self.register_buffer("cos", cos.to(device), persistent=False)
        self.register_buffer("sin", sin.to(device), persistent=False)

        if use_silu_matmul:
            self.layer_norm = RMSNorm(hidden_size, eps=1e-5)
        else:
            self.layer_norm = nn.LayerNorm(hidden_size, device=device)

        self.block_list = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerLayer(
                hidden_size,
                n_heads,
                scale_type,
                device=device,
                use_biases=use_biases,
                interleaved=interleaved,
                model_type=model_type,
                use_silu_matmul=use_silu_matmul,
            )
            self.block_list.append(layer)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply forward pass of the transformer model.

        Args:

            x (torch.Tensor): The input tensor of shape [batch_size, seq_len,
            hidden_size].

            attn_mask (torch.Tensor): The attention mask tensor of shape
            [batch_size, n_heads, seq_len, seq_len].

            k_cache (torch.Tensor): The key cache tensor of shape [batch_size,
            n_layers, max_seq_len, n_heads, head_dim].

            v_cache (torch.Tensor): The value cache tensor of shape [batch_size,
            n_layers, max_seq_len, n_heads, head_dim].

            pos (int): The position of the current layer.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the output tensor of shape [batch_size, seq_len, hidden_size], the
            key cache tensor of shape [batch_size, n_layers, pos+1, n_heads,
            head_dim], and the value cache tensor of shape [batch_size,
            n_layers, pos+1, n_heads, head_dim].
        """
        k_list = []
        v_list = []

        for block_idx, block in enumerate(self.block_list):
            x, k_out, v_out = block(
                x, attn_mask, self.cos, self.sin, k_cache, v_cache, pos, block_idx
            )

            k_list.append(k_out)
            v_list.append(v_out)

        x = self.layer_norm(x)

        # return x, torch.stack(k_list, dim=1), torch.stack(v_list, dim=1)
        return x, k_list, v_list


class LanguageModel(nn.Module):
    """
    A language model that uses a Transformer to generate text.

    Attributes:
        model_type (str): The type of model being used.

        embedding_layer (nn.Linear): The embedding layer for the model.

        transformer (Transformer): The Transformer used by the model.

        logits_layer (nn.Linear): The layer used to compute logits for the
        model.

    Methods:
        get_input_embeddings(): Returns the input embeddings for the model.
        forward(x, attn_mask, k_cache, v_cache, pos): Computes the forward pass
        of the model.
    """

    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        seq_len: int,
        scale_type: str,
        device=torch.device("cpu"),
        model_type: str = "TNLG",
    ) -> None:
        """
        Initialize a LanguageModel object.

        Initializes a LanguageModel object with the given parameters.

        Args:
            n_layers (int): The number of layers in the Transformer.

            vocab_size (int): The size of the vocabulary.

            hidden_size (int): The size of the hidden layer in the Transformer.

            n_heads (int): The number of attention heads in the Transformer.

            seq_len (int): The maximum sequence length.

            scale_type (str): The type of scaling to use in the Transformer.

            device (torch.device, optional): The device to use for computation.
            Defaults to torch.device("cpu").

            model_type (str, optional): The type of model to use. Must be either
            "TNLG" or "Llama". Defaults to "TNLG".

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing
            the logits tensor of shape [batch_size, vocab_size], the key cache
            tensor of shape [batch_size, n_layers, pos+1, n_heads, head_dim],
            and the value cache tensor of shape [batch_size, n_layers, pos+1,
            n_heads, head_dim].

        Raises:
            ValueError: Raised if model_type is not "TNLG", "Llama", or "Dolly".
        """
        super().__init__()
        self.model_type = model_type

        if self.model_type == "TNLG":
            use_biases = True
            interleaved = False
            use_silu_matmul = False
            head_scale = 1.0
            # TNLG uses the logit layer for both embedding the tokens and
            # getting logits.
            self.embedding_layer = None

        elif self.model_type == "Llama":
            use_biases = False
            interleaved = True
            use_silu_matmul = True
            head_scale = 1.0
            self.embedding_layer = nn.Linear(
                hidden_size, vocab_size, bias=False, device=device
            )
        elif self.model_type == "Dolly":
            use_biases = True
            interleaved = False
            use_silu_matmul = False
            head_scale = 0.25
            self.embedding_layer = nn.Linear(
                hidden_size, vocab_size, bias=False, device=device
            )
        else:
            raise ValueError("model_type must be either TNLG or Llama")

        self.transformer = Transformer(
            n_layers,
            hidden_size,
            n_heads,
            seq_len,
            scale_type,
            device=device,
            use_biases=use_biases,
            interleaved=interleaved,
            model_type=model_type,
            head_scale=head_scale,
            use_silu_matmul=use_silu_matmul,
        )
        self.logits_layer = nn.Linear(
            hidden_size, vocab_size, bias=False, device=device
        )

    def get_input_embeddings(self) -> torch.Tensor:
        """
        Get the input embeddings of the model.

        Returns:
            torch.Tensor: The input embeddings of the model.
        """
        if self.model_type == "TNLG":
            return self.logits_layer.weight

        assert self.embedding_layer is not None
        if self.model_type == "Llama" or self.model_type == "Dolly":
            return self.embedding_layer.weight
        else:
            raise ValueError("model_type must be either TNLG or Llama")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

            attn_mask (torch.Tensor): The attention mask tensor.

            k_cache (torch.Tensor): The key cache tensor.

            v_cache (torch.Tensor): The value cache tensor.

            pos (int): The position.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The logits, key
            output, and value output tensors.
        """
        # Dimension of x is [batch_size, seq_len, hidden_size]
        # Dimension of k_cache and v_cache is
        # [batch_size, pos, n_heads, head_dim]
        x, k_out, v_out = self.transformer(x, attn_mask, k_cache, v_cache, pos)

        # Only need logits for the latest token, but we save it and the next
        # token to ease the computation of perplexity.
        # print(x.shape)
        # print(self.logits_layer)
        self.logits = self.logits_layer(x)
        return self.logits, k_out, v_out
        # print(full_logits.shape)
        # exit()
        # self.logits = self.logits_layer(x[:, -1, :])

        # return self.logits, k_out, v_out


class TNLG(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        seq_len: int,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.config = None
        use_biases = True
        interleaved = False
        use_silu_matmul = False
        head_scale = 1.0
        self.embedding_layer = None

        self.transformer = Transformer(
            n_layers,
            hidden_size,
            n_heads,
            seq_len,
            "HeadDim",
            device=device,
            use_biases=use_biases,
            interleaved=interleaved,
            model_type="TNLG",
            head_scale=head_scale,
            use_silu_matmul=use_silu_matmul,
        )
        self.logits_layer = nn.Linear(
            hidden_size, vocab_size, bias=False, device=device
        )

        self.max_seq_len, self.n_layers, self.n_heads, self.hidden_size, self.device = (
            seq_len,
            n_layers,
            n_heads,
            hidden_size,
            device,
        )
        self.head_dim = int(hidden_size / n_heads)

    def generate(self, input_ids, eod, max_length, generation_scale=10.0):
        max_seq_len, n_layers, n_heads, hidden_size, device = (
            self.max_seq_len,
            self.n_layers,
            self.n_heads,
            self.hidden_size,
            self.device,
        )

        tokens = input_ids.to(device)
        x = (
            torch.nn.functional.embedding(
                tokens, self.get_input_embeddings(), None, None, 2.0, False, False
            ).unsqueeze(0)
            * generation_scale
        )
        attn_mask = -10000.0 * torch.triu(
            torch.ones(x.shape[0], max_seq_len, max_seq_len), diagonal=1
        ).to(device)

        head_dim = int(hidden_size / n_heads)
        k_cache = torch.zeros(
            (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
        ).to(device)
        v_cache = torch.zeros(
            (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
        ).to(device)

        pos = 0
        output_tokens = []
        for idx in range(max_length):
            logits, k_out, v_out = self.forward(
                x, attn_mask, k_cache[:, :, :pos], v_cache[:, :, :pos], pos
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == eod:
                break
            seq_len = x.shape[1]

            pruned_size = k_out.shape[-2]
            k_cache[:, :, pos : pos + seq_len, :pruned_size] = k_out
            v_cache[:, :, pos : pos + seq_len, :pruned_size] = v_out
            pos = pos + seq_len
            x = (
                torch.nn.functional.embedding(
                    next_token,
                    self.get_input_embeddings(),
                    None,
                    None,
                    2.0,
                    False,
                    False,
                ).unsqueeze(0)
                * generation_scale
            )
            x = x.reshape(1, 1, hidden_size)
            output_tokens.extend(next_token)
        return output_tokens

    def get_input_embeddings(self) -> torch.Tensor:
        return self.logits_layer.weight

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        return {"input_ids": input_ids}

    def forward(
        self,
        x: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attn_mask is None:
            max_seq_len, n_layers, n_heads, hidden_size, device = (
                self.max_seq_len,
                self.n_layers,
                self.n_heads,
                self.hidden_size,
                self.device,
            )
            scale = 10.0
            tokens = x.to(device)
            x = (
                torch.nn.functional.embedding(
                    tokens, self.get_input_embeddings(), None, None, 2.0, False, False
                )
                * scale
            )
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            attn_mask = -10000.0 * torch.triu(
                torch.ones(x.shape[0], max_seq_len, max_seq_len), diagonal=1
            ).to(device)

            head_dim = int(hidden_size / n_heads)
            k_cache = torch.zeros(
                (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
            ).to(device)[:, :, :pos]
            v_cache = torch.zeros(
                (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
            ).to(device)[:, :, :pos]

        x, k_out, v_out = self.transformer(x, attn_mask, k_cache, v_cache, pos)
        self.logits = self.logits_layer(x)
        return self.logits, k_out, v_out


class CausalLM(nn.Module):
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        hidden_size: int,
        n_heads: int,
        seq_len: int,
        device=torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.config = None
        use_biases = True
        interleaved = False
        use_silu_matmul = False
        head_scale = 1.0
        self.embedding_layer = None

        self.transformer = Transformer(
            n_layers,
            hidden_size,
            n_heads,
            seq_len,
            "HeadDim",
            device=device,
            use_biases=use_biases,
            interleaved=interleaved,
            model_type="TNLG",
            head_scale=head_scale,
            use_silu_matmul=use_silu_matmul,
        )
        self.logits_layer = nn.Linear(
            hidden_size, vocab_size, bias=False, device=device
        )

        self.max_seq_len, self.n_layers, self.n_heads, self.hidden_size, self.device = (
            seq_len,
            n_layers,
            n_heads,
            hidden_size,
            device,
        )
        self.head_dim = int(hidden_size / n_heads)

    def generate(self, input_ids, eod, max_length, generation_scale=10.0):
        max_seq_len, n_layers, n_heads, hidden_size, device = (
            self.max_seq_len,
            self.n_layers,
            self.n_heads,
            self.hidden_size,
            self.device,
        )

        tokens = input_ids.to(device)
        x = (
            torch.nn.functional.embedding(
                tokens, self.get_input_embeddings(), None, None, 2.0, False, False
            ).unsqueeze(0)
            * generation_scale
        )
        attn_mask = -10000.0 * torch.triu(
            torch.ones(x.shape[0], max_seq_len, max_seq_len), diagonal=1
        ).to(device)

        head_dim = int(hidden_size / n_heads)
        k_cache = torch.zeros(
            (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
        ).to(device)
        v_cache = torch.zeros(
            (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
        ).to(device)

        pos = 0
        output_tokens = []
        for idx in range(max_length):
            logits, k_out, v_out = self.forward(
                x, attn_mask, k_cache[:, :, :pos], v_cache[:, :, :pos], pos
            )
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == eod:
                break
            seq_len = x.shape[1]

            pruned_size = k_out.shape[-2]
            k_cache[:, :, pos : pos + seq_len, :pruned_size] = k_out
            v_cache[:, :, pos : pos + seq_len, :pruned_size] = v_out
            pos = pos + seq_len
            x = (
                torch.nn.functional.embedding(
                    next_token,
                    self.get_input_embeddings(),
                    None,
                    None,
                    2.0,
                    False,
                    False,
                ).unsqueeze(0)
                * generation_scale
            )
            x = x.reshape(1, 1, hidden_size)
            output_tokens.extend(next_token)
        return output_tokens

    def get_input_embeddings(self) -> torch.Tensor:
        return self.logits_layer.weight

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs):
        return {"input_ids": input_ids}

    def forward(
        self,
        x: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
        pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attn_mask is None:
            max_seq_len, n_layers, n_heads, hidden_size, device = (
                self.max_seq_len,
                self.n_layers,
                self.n_heads,
                self.hidden_size,
                self.device,
            )
            scale = 10.0
            tokens = x.to(device)
            x = (
                torch.nn.functional.embedding(
                    tokens, self.get_input_embeddings(), None, None, 2.0, False, False
                )
                * scale
            )
            if len(x.shape) == 2:
                x = x.unsqueeze(0)

            attn_mask = -10000.0 * torch.triu(
                torch.ones(x.shape[0], max_seq_len, max_seq_len), diagonal=1
            ).to(device)

            head_dim = int(hidden_size / n_heads)
            k_cache = torch.zeros(
                (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
            ).to(device)[:, :, :pos]
            v_cache = torch.zeros(
                (x.shape[0], n_layers, max_seq_len, n_heads, head_dim)
            ).to(device)[:, :, :pos]

        x, k_out, v_out = self.transformer(x, attn_mask, k_cache, v_cache, pos)
        self.logits = self.logits_layer(x)
        return self.logits, k_out, v_out
