# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rollback-safe draft Engram lookup sharing the target's PLE table."""

import torch


def gather_ngram_context(
    all_token_ids: torch.Tensor,
    request_indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    context_length: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Read only committed positions strictly before each new anchor."""
    offsets = torch.arange(-context_length, 0, device=all_token_ids.device)
    positions = anchor_positions.unsqueeze(1) + offsets
    valid = (request_indices.unsqueeze(1) >= 0) & (positions >= 0)
    valid &= positions < all_token_ids.shape[1]
    tokens = all_token_ids[
        request_indices.clamp_min(0).long().unsqueeze(1),
        positions.clamp(0, all_token_ids.shape[1] - 1).long(),
    ]
    return torch.where(valid, tokens, eos_token_id)


class DSparkNgramLookup:
    def __init__(self, target_model, hidden_size: int) -> None:
        target = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        layers = target.model.layers
        ple = getattr(layers[1], "ple", None)
        if ple is None:
            raise ValueError("DSpark Engram requires the target PLE at layer 2")
        self.lookup = ple.ple_embedding
        if self.lookup.embedding_dim != hidden_size:
            raise ValueError("Target raw Engram width does not match the draft")
        self.context_length = self.lookup.ngram_size - 1
        self.eos_token_id = self.lookup.eos_token_id
        if self.context_length < 1:
            raise ValueError("Target Engram requires ngram_size >= 2")

    def __call__(
        self, previous: torch.Tensor, context: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        starts = torch.arange(
            previous.numel() + 1, device=previous.device, dtype=torch.int32
        )
        ids = self.lookup.compute_ngram_ids(previous, starts, context)
        table = self.lookup.ngram_embedding
        # Never touch the target's asynchronous prefetch buffer/stream.
        values = table.sync_lookup(ids) if table.supports_prefetch else table(ids)
        return table.dequantize(values.flatten(-2), dtype).to(dtype)
