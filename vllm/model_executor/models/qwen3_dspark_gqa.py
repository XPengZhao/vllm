# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecForge context-only Markov GQA/MHA with an independent paged cache."""

import torch
from torch import nn

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

from .qwen3_dspark import DSparkMarkovHead


def validate_markov_gqa_config(vllm_config, config):
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ):
        raise ValueError("Markov GQA currently requires TP=PP=CP=1")
    draft = vllm_config.speculative_config.draft_model_config
    if draft.dtype != torch.bfloat16 or draft.quantization is not None:
        raise ValueError("Markov GQA currently requires unquantized BF16")
    if vllm_config.cache_config.cache_dtype != "auto":
        raise ValueError("Markov GQA requires cache_dtype=auto")
    if (
        getattr(config, "dspark_draft_topk", None) is not None
        or not getattr(config, "sample_from_anchor", True)
        or (getattr(config, "draft_vocab_size", None) or config.vocab_size)
        != config.vocab_size
    ):
        raise ValueError("Markov GQA requires full-vocab anchor sampling without top-k")
    if (
        getattr(config, "attention_bias", False)
        or not getattr(config, "is_neox_style", True)
        or getattr(config, "is_causal", False)
        or getattr(config, "use_sliding_window", False)
        or (getattr(config, "dflash_config", None) or {}).get("causal", False)
        or (getattr(config, "dflash_config", None) or {}).get("use_swa", False)
        or any(
            kind != "full_attention"
            for kind in (getattr(config, "layer_types", None) or [])
        )
    ):
        raise ValueError(
            "Markov GQA requires bias-free non-causal full attention and NeoX RoPE"
        )


class Qwen3MarkovGQA(nn.Module):
    """W1 -> Q/K-normalized attention over target context -> O -> W2."""

    def __init__(self, config, cache_config, prefix):
        super().__init__()
        head = DSparkMarkovHead(
            config.vocab_size, config.vocab_size, config.markov_rank, prefix
        )
        self.markov_w1 = head.markov_w1
        self.markov_w2 = head.markov_w2
        options = (getattr(config, "dflash_config", None) or {}).get("markov_gqa", {})
        self.num_heads = options.get("num_attention_heads", config.num_attention_heads)
        self.num_kv_heads = options.get(
            "num_key_value_heads", config.num_key_value_heads
        )
        self.head_dim = options.get("head_dim", config.head_dim)
        for name in ("num_heads", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"Markov GQA {name} must be a positive integer")
        if self.num_heads % self.num_kv_heads or self.head_dim % 2:
            raise ValueError(
                "Markov GQA requires Q heads divisible by KV heads and even head_dim"
            )

        def linear(name, input_size, output_size):
            return ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                return_bias=False,
                prefix=f"{prefix}.{name}",
            )

        self.q_proj = linear(
            "q_proj", config.markov_rank, self.num_heads * self.head_dim
        )
        self.k_proj = linear(
            "k_proj", config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.v_proj = linear(
            "v_proj", config.hidden_size, self.num_kv_heads * self.head_dim
        )
        self.o_proj = linear(
            "o_proj", self.num_heads * self.head_dim, config.markov_rank
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=config.rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttentionBackend,
        )

    def embed(self, token_ids):
        return self.markov_w1(token_ids)

    def project_query(self, positions, embedding):
        query = self.q_norm(
            self.q_proj(embedding).view(-1, self.num_heads, self.head_dim)
        )
        query, _ = self.rotary_emb(positions, query.flatten(1), None)
        return query

    def project_kv(self, positions, context):
        key = self.k_norm(
            self.k_proj(context).view(-1, self.num_kv_heads, self.head_dim)
        )
        key, _ = self.rotary_emb(positions, key.flatten(1), None)
        value = self.v_proj(context).view(-1, self.num_kv_heads, self.head_dim)
        return key.view(-1, self.num_kv_heads, self.head_dim), value

    def bias(self, embedding, positions, logits_processor):
        query = self.project_query(positions, embedding)
        output = self.attn(query, None, None)
        return logits_processor(self.markov_w2, self.o_proj(output))
