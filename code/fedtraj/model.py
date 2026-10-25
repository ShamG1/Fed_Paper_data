"""Shared trajectory predictor: gated TCN-GRU encoder + low-rank future queries + Transformer decoder.

Same structure as the adopted backbone of the previous study (``src_fed/Mine_Fed_Mine_Transformer_improve.py``
with ``use_residual=False``), with two interface changes:

* the input is the 30 observed frames of physically meaningful, globally standardized features
  (``data.FeatureTransform``) instead of 29 per-window min-max scaled differences;
* the output is a residual relative to constant-velocity extrapolation, in units of the per-step
  residual scale (``data.FeatureTransform.res_scale``).  ``data`` adds the CV displacement and multiplies by
  the scale, so the network starts from the kinematic solution and every loss is measured in meters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model=128, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                      # x: [B, T, D]
        residual = x
        out = x.transpose(1, 2)
        out = self.dropout(self.relu(self.norm1(self.conv1(out))))
        out = self.dropout(self.norm2(self.conv2(out)))
        return self.relu(out.transpose(1, 2) + residual)


class TCNExpert(nn.Module):
    def __init__(self, d_model=128, num_layers=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(*[TemporalConvBlock(d_model, 3, 2 ** i, dropout) for i in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.out_norm(self.net(x))


class GRUExpert(nn.Module):
    def __init__(self, d_model=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        return self.out_norm(out)


class GatedTCNGRUEncoder(nn.Module):
    def __init__(self, d_model=128, tcn_layers=3, gru_layers=1, dropout=0.1):
        super().__init__()
        self.tcn_expert = TCNExpert(d_model, tcn_layers, dropout)
        self.gru_expert = GRUExpert(d_model, gru_layers, dropout)
        self.gate_net = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.fuse_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):                     # [B, T, D] -> memory [T, B, D]
        h_tcn = self.tcn_expert(src)
        h_gru = self.gru_expert(src)
        gate = self.gate_net(torch.cat([h_tcn, h_gru], dim=-1))
        fused = self.dropout(self.fuse_norm(gate * h_tcn + (1.0 - gate) * h_gru + src))
        return fused.transpose(0, 1)


class LowRankFutureTokens(nn.Module):
    def __init__(self, pred_len=50, d_model=128, query_rank=16):
        super().__init__()
        self.time_coeff = nn.Parameter(torch.randn(pred_len, query_rank) * 0.02)
        self.motion_basis = nn.Parameter(torch.randn(query_rank, d_model) * 0.02)
        self.future_pos_embed = nn.Parameter(torch.randn(1, pred_len, d_model) * 0.02)

    def forward(self, batch_size):
        q = (self.time_coeff @ self.motion_basis).unsqueeze(0) + self.future_pos_embed
        return q.expand(batch_size, -1, -1)


class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size=44, obs_len=30, pred_len=50, output_size=2, d_model=128, nhead=4,
                 num_decoder_layers=2, dim_feedforward=256, dropout=0.1, query_rank=16, tcn_layers=3,
                 gru_layers=1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.obs_pos_embed = nn.Parameter(torch.randn(1, obs_len, d_model) * 0.02)
        self.encoder = GatedTCNGRUEncoder(d_model, tcn_layers, gru_layers, dropout)
        self.future_tokens = LowRankFutureTokens(pred_len, d_model, query_rank)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                           dropout=dropout, activation="relu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.output_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(d_model, output_size))
        # start exactly at the CV solution: zero residual at initialization
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, x):                       # x: [B, obs_len, input_size] -> residual [B, pred_len, 2]
        src = self.input_proj(x) + self.obs_pos_embed[:, :x.size(1)]
        memory = self.encoder(src)
        tgt = self.future_tokens(x.size(0)).transpose(0, 1)
        decoded = self.decoder(tgt=tgt, memory=memory).transpose(0, 1)
        return self.output_head(decoded)


def build_model(cfg=None):
    return TrajectoryPredictor(**(cfg or {}))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
