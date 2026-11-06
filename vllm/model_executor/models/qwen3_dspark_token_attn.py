# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3 token-history Markov MHA over committed prefix and private block KV."""

from copy import deepcopy
from typing import cast

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from .qwen3_dspark_gqa import Qwen3MarkovGQA


class OutputNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))
        self.eps = eps

    def forward(self, x):
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return values.to(x.dtype) * self.weight + self.bias


class Qwen3TokenHistoryAttention(Qwen3MarkovGQA):
    options_key = "markov_token_attention"

    def __init__(self, config, cache_config, prefix):
        cfg = deepcopy(config)
        cfg.hidden_size = config.markov_rank
        super().__init__(cfg, cache_config, prefix)
        if self.num_heads != self.num_kv_heads:
            raise ValueError("Token-history Markov attention requires MHA")
        self.output_norm = OutputNorm(config.markov_rank, config.rms_norm_eps)

    def store_prefix(self, ids, positions, slots):
        # Rejected and graph padding rows have no writable slot or valid token ID.
        embedding = self.embed(torch.where(slots >= 0, ids, 0))
        key, value = self.project_kv(positions, embedding)
        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, key, value, self.attn.kv_cache, slots)

    def step_bias(
        self, embedding, positions, local_keys, local_values, logits_processor
    ):
        query = self.project_query(positions, embedding).reshape(
            -1, self.num_heads, self.head_dim
        )
        key, value = self.project_kv(positions, embedding)
        local_keys.append(key)
        local_values.append(value)
        # One KV per known input, including the current token. These lists are
        # local to one proposal, so a rejected suffix never enters shared pages.
        keys = torch.stack(local_keys, dim=2)
        values = torch.stack(local_values, dim=2)
        scores = (keys.float() * query.float().unsqueeze(2)).sum(
            -1
        ) * self.head_dim**-0.5

        if get_forward_context().attn_metadata is not None:
            from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

            metadata, _, cache, _ = get_attention_context(self.attn.layer_name)
            kc, vc = cache.transpose(1, 2).split(self.head_dim, dim=-1)
            impl = cast(FlashAttentionImpl, self.attn.impl)
            fa_version = impl.vllm_flash_attn_version
            if fa_version is None:
                raise RuntimeError(
                    "Token-history attention requires CUDA FlashAttention"
                )
            prefix, lse = flash_attn_varlen_func(
                q=query,
                k=kc,
                v=vc,
                cu_seqlens_q=metadata.query_start_loc,
                seqused_k=metadata.seq_lens,
                max_seqlen_q=1,
                max_seqlen_k=metadata.max_seq_len,
                block_table=metadata.block_table,
                softmax_scale=self.head_dim**-0.5,
                causal=False,
                return_softmax_lse=True,
                num_splits=1,
                fa_version=fa_version,
            )
            # The prefix contributes exp(LSE) mass to the joint softmax.
            visible = metadata.seq_lens[:, None, None] > 0
            prefix_score = torch.where(visible, lse.T.unsqueeze(-1), -torch.inf)
            scores = torch.cat((prefix_score, scores), dim=2)
            values = torch.cat(
                (torch.where(visible, prefix, 0).unsqueeze(2), values), dim=2
            )
        attended = (
            (scores.softmax(-1).unsqueeze(-1) * values.float()).sum(2).to(query.dtype)
        )
        latent = self.output_norm(embedding + self.o_proj(attended.flatten(1)))
        return logits_processor(self.markov_w2, latent)
