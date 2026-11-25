# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecForge's causal bigram/trigram memory for sequential DSpark sampling."""

from itertools import accumulate

import torch
from torch import nn


class DSparkEngramMemory(nn.Module):
    def __init__(self, config, options):
        super().__init__()
        if options["hash_version"] != "xor_v1":
            raise ValueError("Unsupported Markov Engram hash version")
        self.vector_gate = options.get("vector_gate", False)
        if type(self.vector_gate) is not bool:
            raise ValueError("Engram vector_gate must be a boolean")
        self.heads = options.get("heads_per_order", 2)
        self.counts = tuple(options["bucket_counts"])
        self.multipliers = tuple(options["hash_multipliers"])
        dim = options["head_dim"]
        if (
            type(self.heads) is not int
            or self.heads <= 0
            or len(self.counts) != 2 * self.heads
            or any(type(n) is not int or n < 2 for n in self.counts)
            or len(set(self.counts)) != len(self.counts)
            or type(dim) is not int
            or dim <= 0
            or len(self.multipliers) != 3
            or any(type(n) is not int or n <= 0 or n % 2 == 0 for n in self.multipliers)
        ):
            raise ValueError("Invalid Markov Engram table or hash configuration")
        width = len(self.counts) * dim
        self.table = nn.Embedding(sum(self.counts), dim)
        self.register_buffer("moduli", torch.tensor(self.counts), persistent=False)
        self.register_buffer(
            "offsets",
            torch.tensor([0, *accumulate(self.counts[:-1])]),
            persistent=False,
        )
        self.query = nn.Linear(config.hidden_size, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, config.markov_rank, bias=False)
        self.query_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.key_norm = nn.RMSNorm(width, eps=config.rms_norm_eps)
        self.scale = width**-0.5
        if self.vector_gate:
            self.vector_gate_norm = nn.RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.vector_gate_proj = nn.Linear(config.hidden_size, config.markov_rank)

    def modulation(self, hidden_states):
        if not self.vector_gate:
            return None
        return (
            2.0
            * self.vector_gate_proj(self.vector_gate_norm(hidden_states))
            .float()
            .sigmoid()
        )

    def pack_weights(self, weights, prefix):
        rows = []
        for idx, count in enumerate(self.counts):
            name = f"{prefix}tables.{idx}.weight"
            row = weights.pop(name, None)
            if row is None or row.shape != (count, self.table.embedding_dim):
                raise ValueError(f"Missing or incompatible Engram table: {name}")
            rows.append(row)
        weights[prefix + "table.weight"] = torch.cat(rows)
        for name, parameter in self.named_parameters():
            if prefix + name not in weights:
                raise ValueError(f"Missing Engram weight: {prefix}{name}")
            if weights[prefix + name].shape != parameter.shape:
                raise ValueError(f"Incompatible Engram weight: {prefix}{name}")

    def correction(self, query, older, previous, current, modulation=None):
        m0, m1, m2 = self.multipliers
        bigram = current.long() * m0 ^ (previous * m1)
        trigram = bigram ^ (older * m2)
        hashes = torch.stack((bigram, trigram), dim=-1).repeat_interleave(
            self.heads, dim=-1
        )
        ids = hashes.remainder(self.moduli) + self.offsets
        valid2 = previous >= 0
        valid = torch.stack((valid2, valid2 & (older >= 0)), dim=-1)
        memory = self.table(ids).unflatten(1, (2, self.heads))
        memory = (memory * valid[:, :, None, None]).flatten(1)
        key = self.key_norm(self.key(memory))
        gate = (
            (query.float() * key.float()).sum(-1, keepdim=True) * self.scale
        ).sigmoid()
        value = self.value(memory)
        if self.vector_gate:
            if modulation is None:
                raise ValueError("Vector-gated Engram requires hidden-state modulation")
            value = modulation.to(value.dtype) * value
        return gate.to(value.dtype) * value


def engram_history(all_token_ids, request_rows, anchor_positions):
    """Take only the two committed predecessors; -1 disables missing orders."""
    rows = request_rows.long().clamp_min(0)

    def token_at(offset):
        pos = anchor_positions - offset
        ids = all_token_ids[rows, pos.clamp(0, all_token_ids.shape[1] - 1)]
        return torch.where((request_rows >= 0) & (pos >= 0), ids, -1).long()

    return token_at(2), token_at(1)
