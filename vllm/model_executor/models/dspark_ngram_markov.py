# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hidden-conditioned Engram residual used by SpecForge's Markov head."""

import math

import torch
from torch import nn


class DSparkNgramMarkovAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int, eps: float) -> None:
        super().__init__()
        self.ngram_hidden_norm = nn.RMSNorm(
            hidden_size, eps=eps, elementwise_affine=False
        )
        self.ngram_query_proj = nn.Linear(hidden_size, rank, bias=False)
        self.ngram_query_norm = nn.RMSNorm(rank, eps=eps, elementwise_affine=False)
        self.ngram_norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.ngram_key_proj = nn.Linear(hidden_size, rank, bias=False)
        self.ngram_key_norm = nn.RMSNorm(rank, eps=eps, elementwise_affine=False)
        self.ngram_value_proj = nn.Linear(hidden_size, rank, bias=False)
        nn.init.zeros_(self.ngram_value_proj.weight)

    def forward(
        self,
        unigram: torch.Tensor,
        hidden: torch.Tensor,
        ngram: torch.Tensor,
    ) -> torch.Tensor:
        hidden = hidden.to(unigram.dtype)
        ngram = self.ngram_norm(ngram.to(unigram.dtype))
        query = self.ngram_query_norm(
            self.ngram_query_proj(self.ngram_hidden_norm(hidden))
        )
        key = self.ngram_key_norm(self.ngram_key_proj(ngram))
        value = self.ngram_value_proj(ngram)
        score = (query.float() * key.float()).sum(-1, keepdim=True)
        score = score / math.sqrt(query.shape[-1])
        gate = torch.sigmoid(
            torch.sign(score) * torch.sqrt(score.abs().clamp_min(1e-6))
        ).to(value.dtype)
        return unigram + gate * value
