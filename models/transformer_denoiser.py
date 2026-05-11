from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8000) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ConvFrontend(nn.Module):
    """Conv1D along frequency axis to capture harmonic structure per time frame."""

    def __init__(self, freq_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        x = x.reshape(B * T, 1, F)
        x = self.net(x)
        return x.reshape(B, T, F)


class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        causal: bool = False,
        complex_mask: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.complex_mask = complex_mask

        # --- freq expansion for compressed features (mel / rnnoise-like) ---
        if input_dim != output_dim:
            self.freq_expand = nn.Sequential(
                nn.Linear(input_dim, (input_dim + output_dim) // 2),
                nn.ReLU(),
                nn.Linear((input_dim + output_dim) // 2, output_dim),
                nn.LayerNorm(output_dim),
            )
            feat_dim = output_dim
        else:
            self.freq_expand = None
            feat_dim = input_dim

        # --- conv frontend: harmonic structure along frequency axis ---
        self.conv_frontend = ConvFrontend(feat_dim)

        # --- transformer trunk ---
        self.input_proj = nn.Linear(feat_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=8000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- output: magnitude mask or complex mask ---
        if complex_mask:
            self.output_proj = nn.Linear(d_model, output_dim * 2)
            self.output_activation = nn.Identity()
        else:
            self.output_proj = nn.Linear(d_model, output_dim)
            self.output_activation = nn.Sigmoid()

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.freq_expand is not None:
            x = self.freq_expand(x)

        x = self.conv_frontend(x) + x  # residual connection

        hidden = self.input_proj(x)
        hidden = self.positional_encoding(hidden)

        src_mask = None
        if self.causal:
            src_mask = self._causal_mask(x.size(1), x.device)

        hidden = self.encoder(
            hidden,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        output = self.output_activation(self.output_proj(hidden))
        if self.complex_mask:
            output = output.view(output.shape[0], output.shape[1], -1, 2)
        return output
