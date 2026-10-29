# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SpecForge Qwen3 MLA with expanded, independently paged KV caches."""

from dataclasses import replace

import torch
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)

from .qwen3_dflash import DFlashQwen3DecoderLayer, Qwen3MLP
from .qwen3_dspark import DSparkMarkovHead, Qwen3DSparkModel
from .utils import WeightsMapper, get_draft_quant_config


class Qwen3DSparkMLA(nn.Module):
    def __init__(self, config, cache_config, prefix, *, markov=False):
        super().__init__()
        options = config.dflash_config.get("markov_mla", {}) if markov else {}

        def dimension(name, default):
            value = (
                options.get(name, default)
                if markov
                else getattr(config, "mla_" + name, default)
            )
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"MLA {name} must be a positive integer")
            return value

        self.num_heads = (
            options.get("num_attention_heads", 32)
            if markov
            else (config.num_attention_heads)
        )
        if (
            isinstance(self.num_heads, bool)
            or not isinstance(self.num_heads, int)
            or self.num_heads <= 0
        ):
            raise ValueError("MLA num_attention_heads must be a positive integer")
        self.rank = dimension("kv_lora_rank", 512)
        self.nope_dim = dimension("qk_nope_head_dim", 64)
        self.rope_dim = dimension("qk_rope_head_dim", 64)
        self.v_dim = dimension("v_head_dim", 128)
        self.head_dim = self.nope_dim + self.rope_dim
        if self.rope_dim % 2 or self.v_dim != self.head_dim:
            raise ValueError("MLA requires even RoPE width and V width = QK width")
        width = config.markov_rank if markov else config.hidden_size

        def linear(name, input_size, output_size):
            return ReplicatedLinear(
                input_size,
                output_size,
                bias=False,
                return_bias=False,
                prefix=f"{prefix}.{name}",
            )

        self.q_proj = linear("q_proj", width, self.num_heads * self.head_dim)
        self.kv_a_proj = linear(
            "kv_a_proj", config.hidden_size, self.rank + self.rope_dim
        )
        self.kv_a_norm = RMSNorm(self.rank, eps=config.rms_norm_eps)
        self.kv_b_proj = linear(
            "kv_b_proj", self.rank, self.num_heads * (self.nope_dim + self.v_dim)
        )
        self.o_proj = linear("o_proj", self.num_heads * self.v_dim, width)
        set_default_rope_theta(config, default_theta=1000000)
        self.rotary_emb = get_rope(
            self.rope_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters=config.rope_parameters,
        )
        self.causal = False
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
            attn_backend=FlashAttentionBackend,
        )

    def project_query(self, positions, hidden):
        query = self.q_proj(hidden).view(-1, self.num_heads, self.head_dim)
        content, rope = query.split((self.nope_dim, self.rope_dim), dim=-1)
        rope, _ = self.rotary_emb(positions, rope.contiguous().flatten(1), None)
        return torch.cat((content, rope.view_as(query[..., self.nope_dim :])), -1)

    def project_kv(self, positions, context):
        latent, rope = self.kv_a_proj(context).split((self.rank, self.rope_dim), dim=-1)
        kv = self.kv_b_proj(self.kv_a_norm(latent)).view(
            -1, self.num_heads, self.nope_dim + self.v_dim
        )
        content, value = kv.split((self.nope_dim, self.v_dim), dim=-1)
        rope, _ = self.rotary_emb(positions, rope.contiguous(), None)
        key = torch.cat((content, rope[:, None].expand(-1, self.num_heads, -1)), -1)
        return key, value.contiguous()

    def forward(self, positions, hidden_states):
        query = self.project_query(positions, hidden_states)
        key, value = self.project_kv(positions, hidden_states)
        return self.o_proj(self.attn(query.flatten(1), key, value))


class Qwen3MarkovMLA(Qwen3DSparkMLA):
    def __init__(self, config, cache_config, prefix):
        super().__init__(config, cache_config, prefix, markov=True)
        head = DSparkMarkovHead(
            config.vocab_size, config.vocab_size, config.markov_rank, prefix
        )
        self.markov_w1 = head.markov_w1
        self.markov_w2 = head.markov_w2

    def embed(self, token_ids):
        return self.markov_w1(token_ids)

    def bias(self, embedding, positions, logits_processor):
        query = self.project_query(positions, embedding)
        output = self.attn(query.flatten(1), None, None)
        return logits_processor(self.markov_w2, self.o_proj(output))


class Qwen3DSparkMLADecoderLayer(DFlashQwen3DecoderLayer):
    def __init__(
        self,
        vllm_config,
        *,
        config,
        layer_idx,
        cache_config=None,
        quant_config=None,
        prefix="",
    ):
        nn.Module.__init__(self)
        self.self_attn = Qwen3DSparkMLA(config, cache_config, f"{prefix}.self_attn")
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class Qwen3DSparkMLAModel(Qwen3DSparkModel):
    decoder_layer_cls = Qwen3DSparkMLADecoderLayer
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config, start_layer_id=0, prefix=""):
        config = vllm_config.speculative_config.draft_model_config
        parallel = get_current_vllm_config().parallel_config
        if (
            parallel.tensor_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
        ):
            raise ValueError("Qwen3 DSpark MLA currently requires TP=PP=CP=1")
        if config.dtype != torch.bfloat16 or config.quantization is not None:
            raise ValueError("Qwen3 DSpark MLA currently requires unquantized BF16")
        if get_draft_quant_config(vllm_config) is not None:
            raise ValueError("Qwen3 DSpark MLA does not support quantized projections")
        hf = config.hf_config
        if (
            getattr(hf, "is_causal", False)
            or getattr(hf, "use_sliding_window", False)
            or any(
                kind != "full_attention"
                for kind in (getattr(hf, "layer_types", None) or [])
            )
            or getattr(hf, "attention_bias", False)
            or not getattr(hf, "is_neox_style", True)
        ):
            raise ValueError(
                "Qwen3 DSpark MLA requires non-causal full attention, "
                "bias-free projections and NeoX RoPE"
            )
        if (
            getattr(hf, "dspark_draft_topk", None) is not None
            or not getattr(hf, "sample_from_anchor", True)
            or (getattr(hf, "draft_vocab_size", None) or hf.vocab_size) != hf.vocab_size
        ):
            raise ValueError(
                "Qwen3 DSpark MLA requires full-vocab anchor sampling "
                "with no draft top-k"
            )
        if get_current_vllm_config().cache_config.cache_dtype != "auto":
            raise ValueError("Qwen3 DSpark MLA requires cache_dtype=auto")
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )

    def _build_fused_kv_buffers(self):
        # The reference implementation projects independent latent KV weights.
        pass

    def precompute_and_store_context_kv(
        self, context_states, context_positions, context_slot_mapping=None
    ):
        context = self.hidden_norm(context_states)
        modules = [layer.self_attn for layer in self.layers]
        if isinstance(self.markov_head, Qwen3MarkovMLA):
            modules.append(self.markov_head)
        for index, module in enumerate(modules):
            key, value = module.project_kv(context_positions, context)
            slots = (
                context_slot_mapping[index]
                if isinstance(context_slot_mapping, list)
                else context_slot_mapping
            )
            if slots is not None:
                attn = module.attn
                attn.impl.do_kv_cache_update(attn, key, value, attn.kv_cache, slots)


def markov_attention_metadata(metadata, num_reqs, query_start_loc, anchor_positions):
    """One query per request, reading the same pre-anchor prefix at every step."""
    if not isinstance(metadata, FlashAttentionMetadata):
        raise ValueError("Qwen3 Markov MLA requires the FlashAttention backend")
    return replace(
        metadata,
        num_actual_tokens=num_reqs,
        max_query_len=1,
        query_start_loc=query_start_loc[: num_reqs + 1],
        seq_lens=anchor_positions.to(torch.int32).clamp_min(0),
        block_table=metadata.block_table[:num_reqs],
        slot_mapping=metadata.slot_mapping[:num_reqs],
        causal=False,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        scheduler_metadata=None,
        prefix_scheduler_metadata=None,
        max_num_splits=1,
    )
