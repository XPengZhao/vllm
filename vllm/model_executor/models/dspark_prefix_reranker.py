# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecForge's block-local prefix Transformer and residual candidate scorer.

All parameters are replicated across TP ranks. The short prefix KV belongs to
one draft block, not the paged backbone cache or a persistent request state.
Parameter names and computations match SpecForge's PrefixReranker checkpoint.
"""

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

PrefixKV = tuple[torch.Tensor, torch.Tensor]


class DSparkPrefixReranker(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        block_size: int,
        *,
        width: int = 256,
        num_heads: int = 4,
        top_k: int = 16,
    ) -> None:
        super().__init__()
        if width <= 0 or num_heads <= 0 or width % num_heads:
            raise ValueError(
                "reranker width must be positive and divisible by num_heads"
            )
        if top_k < 1 or block_size < 1:
            raise ValueError("reranker top_k and block_size must be positive")
        self.width, self.num_heads = width, num_heads
        self.block_size, self.top_k = block_size, top_k
        self.token_proj = nn.Linear(embedding_size, width, bias=False)
        self.position_embed = nn.Embedding(block_size, width)
        self.attn_norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.attn_out = nn.Linear(width, width, bias=False)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width)
        )
        self.output_norm = nn.LayerNorm(width)
        self.hidden_proj = nn.Linear(hidden_size, width, bias=False)
        self.query = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU())
        self.candidate_proj = nn.Linear(embedding_size, width, bias=False)
        self.residual_out = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.residual_out.weight)

    def encode(
        self, token_embeddings: torch.Tensor, cache: PrefixKV | None = None
    ) -> tuple[torch.Tensor, PrefixKV]:
        """Encode a causal prefix, or append one token to a block-local KV."""
        leading, length = token_embeddings.shape[:-2], token_embeddings.shape[-2]
        start = 0 if cache is None else cache[0].shape[-2]
        if start + length > self.block_size:
            raise ValueError("prefix KV exceeds one block; reset cache at every round")
        if cache is not None and length != 1:
            raise ValueError("incremental reranker expects exactly one new token")
        x = self.token_proj(token_embeddings).reshape(-1, length, self.width)
        pos = torch.arange(start, start + length, device=x.device)
        x = x + self.position_embed(pos)
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(
                -1, length, self.num_heads, self.width // self.num_heads
            ).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=-2)
            v = torch.cat((cache[1], v), dim=-2)
        # A cached single query reads ALL previous keys. PyTorch's upper-left
        # causal mask for a non-square matrix would incorrectly hide the prefix.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None)
        out = out.transpose(1, 2).reshape(-1, length, self.width)
        x = x + self.attn_out(out)
        x = x + self.ffn(self.ffn_norm(x))
        return self.output_norm(x).reshape(*leading, length, self.width), (k, v)

    def score(
        self,
        baseline_logits: torch.Tensor,
        hidden: torch.Tensor,
        prefix: torch.Tensor,
        candidate_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rerank top-k AFTER the full-vocabulary Markov bias is applied."""
        k = min(self.top_k, baseline_logits.shape[-1])
        _, ids = baseline_logits.topk(k, dim=-1)
        # Keep baseline argmax first, including ties excluded by topk, so zero
        # residual preserves the original greedy choice exactly.
        best = baseline_logits.argmax(dim=-1, keepdim=True)
        best_slot = (ids == best).long().argmax(dim=-1, keepdim=True)
        present = (ids == best).any(dim=-1, keepdim=True)
        best_slot = torch.where(present, best_slot, torch.full_like(best_slot, k - 1))
        first_id = ids[..., :1].clone()
        ids = ids.scatter(-1, best_slot, first_id)
        ids[..., :1] = best
        values = baseline_logits.gather(-1, ids)
        # W2 rows encode candidate tokens; W1 is only for previous-token input.
        keys = self.candidate_proj(F.embedding(ids, candidate_table))
        query = self.residual_out(
            self.query(torch.cat((self.hidden_proj(hidden), prefix), dim=-1))
        )
        delta = (query.unsqueeze(-2) * keys).sum(dim=-1) / math.sqrt(self.width)
        return values.float() + delta.float(), ids

    def sample(
        self,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        embed: Callable[[torch.Tensor], torch.Tensor],
        bias: Callable[[torch.Tensor], torch.Tensor],
        candidate_table: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy draft walk using actual choices, resetting KV at each call."""
        cache = None
        tokens = []
        previous = anchor_ids
        for i in range(base_logits.shape[-2]):
            markov_embed = embed(previous)
            prefix, cache = self.encode(markov_embed.unsqueeze(-2), cache)
            scores, ids = self.score(
                base_logits[..., i, :] + bias(markov_embed),
                hidden[..., i, :],
                prefix.squeeze(-2),
                candidate_table,
            )
            previous = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
            tokens.append(previous)
        if not tokens:
            return anchor_ids.new_empty((*anchor_ids.shape, 0))
        return torch.stack(tokens, dim=-1)
