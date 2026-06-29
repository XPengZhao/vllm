# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen3 import Qwen3ForCausalLM
from .utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)

logger = init_logger(__name__)


_DFLASH_VALID_LAYER_TYPES = frozenset({"full_attention", "sliding_attention"})


def _dflash_trace_enabled() -> bool:
    return os.environ.get("VLLM_DFLASH_TRACE") == "1"


def _dflash_hidden_debug_stats(hidden_states: torch.Tensor) -> tuple[float, float, float]:
    hs = hidden_states.detach().float()
    norm_mean = float(hs.norm(dim=-1).mean().item())
    absmax = float(hs.abs().max().item())
    cos_offdiag_mean = float("nan")
    if hs.shape[0] > 1:
        hs_unit = torch.nn.functional.normalize(hs, dim=-1)
        cos = hs_unit @ hs_unit.T
        offdiag = cos[
            ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
        ]
        cos_offdiag_mean = float(offdiag.mean().item())
    return norm_mean, absmax, cos_offdiag_mean


def _dflash_tail_cos(hidden_states: torch.Tensor) -> float:
    if hidden_states.shape[0] <= 2:
        return float("nan")
    return _dflash_hidden_debug_stats(hidden_states[1:])[2]


def _get_dflash_layer_types(config: Qwen3Config) -> tuple[str, ...]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        return ("full_attention",) * config.num_hidden_layers
    if len(layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"DFlash layer_types length {len(layer_types)} does not match "
            f"num_hidden_layers {config.num_hidden_layers}."
        )
    invalid = set(layer_types) - _DFLASH_VALID_LAYER_TYPES
    if invalid:
        raise ValueError(f"Invalid DFlash layer_type(s): {sorted(invalid)}.")
    if "sliding_attention" in layer_types and not getattr(
        config, "sliding_window", None
    ):
        raise ValueError(
            "DFlash sliding_attention layers require `sliding_window` in config."
        )
    return tuple(layer_types)


class DFlashAttention(Attention):
    """Attention with DFlash-specific KV allocation semantics.

    The compute path keeps the layer's configured sliding window. The KV cache
    spec is widened to full attention because DFlash writes every context KV
    before drafting and cannot evict old context blocks from draft layers.
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if getattr(self, "sliding_window", None) is not None:
            return FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        spec = super().get_kv_cache_spec(vllm_config)
        if isinstance(spec, SlidingWindowSpec):
            return FullAttentionSpec(
                block_size=spec.block_size,
                num_kv_heads=spec.num_kv_heads,
                head_size=spec.head_size,
                head_size_v=getattr(spec, "head_size_v", spec.head_size),
                dtype=spec.dtype,
                kv_quant_mode=spec.kv_quant_mode,
                page_size_padded=spec.page_size_padded,
            )
        return spec

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        debug_enabled = (
            (
                os.environ.get("VLLM_DFLASH_ATTN_VALUE_DEBUG") == "1"
                or _dflash_trace_enabled()
            )
            and getattr(self, "_dflash_in_real_propose", False)
            and not getattr(self, "_logged_dflash_attn_value_debug", False)
        )
        if debug_enabled:
            from vllm.model_executor.layers.attention.attention import (
                get_attention_context,
            )

            attn_metadata, _, kv_cache, layer_slot_mapping = get_attention_context(
                self.layer_name
            )
            context = get_forward_context()
            slot_mapping = context.slot_mapping
            slot_mapping_keys = (
                list(slot_mapping.keys())[:8] if isinstance(slot_mapping, dict) else None
            )
            query_float = query.detach().float()
            key_float = key.detach().float()
            value_float = value.detach().float()
            query_flat = query_float.reshape(query_float.shape[0], -1)
            key_flat = key_float.reshape(key_float.shape[0], -1)
            value_flat = value_float.reshape(value_float.shape[0], -1)
            _, _, query_cos = _dflash_hidden_debug_stats(query_flat)
            _, _, key_cos = _dflash_hidden_debug_stats(key_flat)
            _, _, value_cos = _dflash_hidden_debug_stats(value_flat)
            kv_sample = kv_cache.flatten()[:4096].detach().float()
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            seq_lens = getattr(attn_metadata, "seq_lens", None)
            block_table = getattr(attn_metadata, "block_table", None)
            logger.info(
                "DFlash attention value debug before: layer=%s, "
                "metadata=%s, causal=%s, num_actual_tokens=%s, "
                "max_query_len=%s, max_seq_len=%s, query_start_loc=%s, "
                "seq_lens=%s, block_table_shape=%s, block_table_sample=%s, "
                "q_norm=%.6f, k_norm=%.6f, v_norm=%.6f, "
                "q_absmax=%.6f, k_absmax=%.6f, v_absmax=%.6f, "
                "q_cos=%.6f, q_tail_cos=%.6f, "
                "k_cos=%.6f, v_cos=%.6f, "
                "kv_cache_shape=%s, kv_cache_dtype=%s, "
                "kv_cache_sample_absmax=%.6f, "
                "layer_slot_mapping_shape=%s, layer_slot_mapping_sample=%s, "
                "slot_mapping_keys_sample=%s.",
                self.layer_name,
                type(attn_metadata).__name__,
                getattr(attn_metadata, "causal", None),
                getattr(attn_metadata, "num_actual_tokens", None),
                getattr(attn_metadata, "max_query_len", None),
                getattr(attn_metadata, "max_seq_len", None),
                (
                    query_start_loc.detach().cpu().tolist()
                    if query_start_loc is not None
                    else None
                ),
                (
                    seq_lens.detach().cpu().tolist() if seq_lens is not None else None
                ),
                tuple(block_table.shape) if block_table is not None else None,
                (
                    block_table[:2, :8].detach().cpu().tolist()
                    if block_table is not None and block_table.dim() == 2
                    else None
                ),
                float(query_float.norm(dim=-1).mean().item()),
                float(key_float.norm(dim=-1).mean().item()),
                float(value_float.norm(dim=-1).mean().item()),
                float(query_float.abs().max().item()),
                float(key_float.abs().max().item()),
                float(value_float.abs().max().item()),
                query_cos,
                _dflash_tail_cos(query_flat),
                key_cos,
                value_cos,
                tuple(kv_cache.shape),
                kv_cache.dtype,
                float(kv_sample.abs().max().item()) if kv_sample.numel() else 0.0,
                (
                    tuple(layer_slot_mapping.shape)
                    if layer_slot_mapping is not None
                    else None
                ),
                (
                    layer_slot_mapping[:16].detach().cpu().tolist()
                    if layer_slot_mapping is not None
                    else None
                ),
                slot_mapping_keys,
            )

        output = super().forward(query, key, value, output_shape)

        if debug_enabled:
            self._logged_dflash_attn_value_debug = True
            output_float = output.detach().float()
            logger.info(
                "DFlash attention value debug after: layer=%s, "
                "output_shape=%s, output_norm=%.6f, output_absmax=%.6f, "
                "output_cos=%.6f, output_tail_cos=%.6f.",
                self.layer_name,
                tuple(output.shape),
                float(output_float.norm(dim=-1).mean().item()),
                float(output_float.abs().max().item()),
                _dflash_hidden_debug_stats(output_float.reshape(output.shape[0], -1))[
                    2
                ],
                _dflash_tail_cos(output_float.reshape(output.shape[0], -1)),
            )
        return output


class DFlashQwen3Attention(nn.Module):
    """Attention for DFlash speculative decoding.

    Context KVs are pre-inserted into the KV cache before the forward pass.
    This layer handles only query tokens via standard attention.
    Adapted from Qwen3Attention."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        attention_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.layer_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,  # DFlash has o_proj bias when using attention bias
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = DFlashAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """DFlash attention assumes that the KV cache is already populated
        with the context K/V from the target model's hidden states. This forward op
        computes attention for the query tokens only.
        See also: precompute_and_store_context_kv"""
        qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Per-head RMSNorm
        q_shape, k_shape = q.shape, k.shape
        q = self.q_norm(
            q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
        ).view(q_shape)
        k = self.k_norm(
            k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
        ).view(k_shape)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashQwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        layer_type: str = "full_attention",
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = layer_type
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER
        sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.self_attn = DFlashQwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            sliding_window=sliding_window,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        else:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class DFlashQwen3Model(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)

        drafter_config = getattr(self.config, "eagle_config", {})
        drafter_config.update(getattr(self.config, "dflash_config", {}))

        if drafter_config is not None and "use_aux_hidden_state" in drafter_config:
            self.use_aux_hidden_state = drafter_config["use_aux_hidden_state"]
        else:
            self.use_aux_hidden_state = True

        current_vllm_config = get_current_vllm_config()

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        self.layer_types = _get_dflash_layer_types(self.config)
        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                    config=self.config,
                    layer_type=self.layer_types[layer_idx],
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.sliding_attention_layer_names = {
            layer.self_attn.attn.layer_name
            for layer in self.layers
            if layer.layer_type == "sliding_attention"
        }
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
                input_size=fc_input_size,
                output_size=self.config.hidden_size,
                bias=False,
                params_dtype=vllm_config.model_config.dtype,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "fc"),
                return_bias=False,
            )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _build_fused_kv_buffers(self) -> None:
        """Build fused weight buffers for precompute_and_store_context_kv.

        Must be called after weights are loaded. Stacks the KV-projection
        weights, K-norm weights, and RoPE parameters from every attention
        layer so that precompute_and_store_context_kv can run one fused
        GEMM for all layers at once. Also aliases the weight of the hidden_norm.
        """
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data

        # KV projection weights: [num_layers * 2 * kv_size, hidden_size]
        kv_weights = [a.qkv_proj.weight[a.q_size :] for a in layers_attn]
        self._fused_kv_weight = torch.cat(kv_weights, dim=0)
        if has_bias:
            kv_biases = [a.qkv_proj.bias[a.q_size :] for a in layers_attn]
            self._fused_kv_bias: torch.Tensor | None = torch.cat(kv_biases, dim=0)
        else:
            self._fused_kv_bias = None

        # K-norm weights: list of [head_dim] tensors, one per layer.
        self._k_norm_weights = [a.k_norm.weight.data for a in layers_attn]

        # RoPE parameters
        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        # Validation that RoPE params are the same across all layers
        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All layers must have the same RoPE parameters for DFlash precomputation"

        # Layer metadata
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        # Validation that all layers have the same attention config
        for attn in layers_attn[1:]:
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All layers must have the same attn config for DFlash precomputation"

        # References to inner Attention layers for direct cache writes
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute K/V for context states write them into each layer's KV cache.

        Input context states are projected to K/V, normed, and have RoPE applied.
        Since the context shape is different than the query shape, we can't rely on the
        regular forward pass to apply torch.compile and CUDA graphs to this section.
        As such, this function is optimized to minimize the number of torch ops present:
        we use fused vLLM kernels for RMSNorm and RoPE, fuse the GEMM into one
        large projection, and avoid cloning buffers (with .contiguous()) where possible.

        When context_slot_mapping is None (e.g. during dummy_run) only
        the computation runs, and no K/V is written to cache.
        """
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "DFlash buffer initialization was skipped. If dummy weights are not "
                "in use, this may indicate an error in weight loading."
            )
            self._build_fused_kv_buffers()

        num_ctx = context_states.shape[0]
        L = self._num_attn_layers
        kv = self._kv_size
        hd = self._head_dim
        nkv = self._num_kv_heads

        # --- Fused KV projection (one GEMM for all layers) ---
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        # Single contiguous copy that separates K/V and transposes to
        # layer-major layout.  Result: [2, L, num_ctx, nkv, hd] contiguous.
        # Indexing dim-0 gives contiguous [L, num_ctx, nkv, hd] for K and V.
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd], contiguous
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd], contiguous

        # --- Per-layer RMSNorm K (3D: [num_ctx, nkv, hd] per layer) ---
        all_k_normed = torch.empty_like(all_k)
        for i in range(L):
            ops.rms_norm(
                all_k_normed[i],
                all_k[i],
                self._k_norm_weights[i],
                self._rms_norm_eps,
            )

        # --- Fused RoPE across all layers ---
        # View as [L * num_ctx, kv] so RoPE sees one big batch (no copy).
        # In-place RoPE: pass K as the "query" arg with key=None.
        all_k_flat = all_k_normed.view(L * num_ctx, kv)
        positions_repeated = context_positions.repeat(L)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )
        all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)

        if (
            _dflash_trace_enabled()
            and context_slot_mapping is not None
            and not getattr(self, "_logged_dflash_context_kv_trace", False)
        ):
            self._logged_dflash_context_kv_trace = True
            layer_stats = []
            for i in range(min(L, 5)):
                k_norm, k_absmax, k_cos = _dflash_hidden_debug_stats(
                    all_k_final[i].reshape(num_ctx, -1)
                )
                v_norm, v_absmax, v_cos = _dflash_hidden_debug_stats(
                    all_v[i].reshape(num_ctx, -1)
                )
                layer_stats.append(
                    {
                        "layer": i,
                        "k_norm": k_norm,
                        "k_absmax": k_absmax,
                        "k_cos": k_cos,
                        "v_norm": v_norm,
                        "v_absmax": v_absmax,
                        "v_cos": v_cos,
                    }
                )

            ref_max_diff = float("nan")
            ref_v_max_diff = float("nan")
            ref_k_cos = float("nan")
            if num_ctx > 0:
                attn0 = self.layers[0].self_attn
                ref_qkv = F.linear(
                    normed_context_states,
                    attn0.qkv_proj.weight,
                    attn0.qkv_proj.bias,
                )
                ref_k = ref_qkv[:, attn0.q_size : attn0.q_size + attn0.kv_size]
                ref_v = ref_qkv[:, attn0.q_size + attn0.kv_size :]
                ref_k_shape = ref_k.shape
                ref_k = attn0.k_norm(
                    ref_k.view(
                        *ref_k_shape[:-1],
                        ref_k_shape[-1] // attn0.head_dim,
                        attn0.head_dim,
                    )
                ).view(ref_k_shape)
                ref_k, _ = attn0.rotary_emb(context_positions, ref_k, None)
                ref_k = ref_k.view(num_ctx, nkv, hd)
                ref_v = ref_v.view(num_ctx, nkv, hd)
                ref_max_diff = float(
                    (ref_k.float() - all_k_final[0].float()).abs().max().item()
                )
                ref_v_max_diff = float(
                    (ref_v.float() - all_v[0].float()).abs().max().item()
                )
                ref_k_cos = _dflash_hidden_debug_stats(ref_k.reshape(num_ctx, -1))[2]

            logger.info(
                "DFlash trace context KV: num_ctx=%d, positions_tail=%s, "
                "rope_head_size=%d, rope_rotary_dim=%s, rope_is_neox=%s, "
                "fused_vs_ref_k_absmax_diff=%.6f, "
                "fused_vs_ref_v_absmax_diff=%.6f, ref_k_cos=%.6f, "
                "layer_stats=%s.",
                num_ctx,
                context_positions[-min(num_ctx, 16) :].detach().cpu().tolist(),
                self._rope_head_size,
                getattr(self.layers[0].self_attn.rotary_emb, "rotary_dim", None),
                self._rope_is_neox,
                ref_max_diff,
                ref_v_max_diff,
                ref_k_cos,
                layer_stats,
            )

        if context_slot_mapping is None:
            return

        # --- Per-layer cache insert ---
        for i in range(L):
            attn = self._attn_layers[i]
            layer_slot_mapping = (
                context_slot_mapping[attn.layer_name]
                if isinstance(context_slot_mapping, Mapping)
                else context_slot_mapping
            )
            kv_cache = attn.kv_cache
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                kv_cache,
                layer_slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        hidden_states = input_embeds
        debug_env_enabled = (
            os.environ.get("VLLM_DFLASH_LAYER_DEBUG") == "1"
            or os.environ.get("VLLM_DFLASH_ATTN_VALUE_DEBUG") == "1"
            or _dflash_trace_enabled()
        )
        in_real_propose = (
            getattr(self, "_dflash_in_real_propose", False)
            if debug_env_enabled
            else False
        )
        if debug_env_enabled:
            for layer in self.layers:
                setattr(
                    layer.self_attn.attn,
                    "_dflash_in_real_propose",
                    in_real_propose,
                )

        residual = None
        debug_layer_stats = []
        debug_enabled = (
            debug_env_enabled
            and (
                os.environ.get("VLLM_DFLASH_LAYER_DEBUG") == "1"
                or _dflash_trace_enabled()
            )
            and in_real_propose
            and not getattr(self, "_logged_dflash_layer_debug", False)
            and hidden_states.shape[0] > 1
        )
        if debug_enabled:
            norm_mean, absmax, cos_mean = _dflash_hidden_debug_stats(hidden_states)
            debug_layer_stats.append(
                {
                    "stage": "embed",
                    "norm": norm_mean,
                    "absmax": absmax,
                    "cos": cos_mean,
                    "tail_cos": _dflash_tail_cos(hidden_states),
                }
            )

        for layer_idx, layer in enumerate(self.layers):
            if debug_enabled:
                layer_input = hidden_states
                if residual is not None:
                    attn_input, layer_residual = layer.input_layernorm(
                        layer_input, residual
                    )
                else:
                    layer_residual = layer_input
                    attn_input = layer.input_layernorm(layer_input)
                attn_output = layer.self_attn(
                    positions=positions,
                    hidden_states=attn_input,
                )
                norm_mean, absmax, cos_mean = _dflash_hidden_debug_stats(attn_output)
                debug_layer_stats.append(
                    {
                        "stage": f"layer{layer_idx}.attn",
                        "norm": norm_mean,
                        "absmax": absmax,
                        "cos": cos_mean,
                        "tail_cos": _dflash_tail_cos(attn_output),
                    }
                )
                hidden_states, residual = layer.post_attention_layernorm(
                    attn_output, layer_residual
                )
                hidden_states = layer.mlp(hidden_states)
                norm_mean, absmax, cos_mean = _dflash_hidden_debug_stats(hidden_states)
                debug_layer_stats.append(
                    {
                        "stage": f"layer{layer_idx}.mlp",
                        "norm": norm_mean,
                        "absmax": absmax,
                        "cos": cos_mean,
                        "tail_cos": _dflash_tail_cos(hidden_states),
                    }
                )
                continue
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_enabled:
            norm_mean, absmax, cos_mean = _dflash_hidden_debug_stats(hidden_states)
            debug_layer_stats.append(
                {
                    "stage": "final_norm",
                    "norm": norm_mean,
                    "absmax": absmax,
                    "cos": cos_mean,
                    "tail_cos": _dflash_tail_cos(hidden_states),
                }
            )
            self._logged_dflash_layer_debug = True
            logger.info("DFlash trace layer: %s", debug_layer_stats)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.has_own_lm_head = False
        self.has_own_embed_tokens = False
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
        self.target_vocab_size = vllm_config.model_config.get_vocab_size()
        self.uses_draft_vocab_for_input_ids = (
            self.model.embed_tokens.org_vocab_size == self.config.draft_vocab_size
        )
        if self.config.draft_vocab_size != self.target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
            self.target_id_to_draft_id = nn.Parameter(
                torch.zeros(self.target_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None
            self.target_id_to_draft_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            os.environ.get("VLLM_DFLASH_LAYER_DEBUG") == "1"
            or os.environ.get("VLLM_DFLASH_ATTN_VALUE_DEBUG") == "1"
            or _dflash_trace_enabled()
        ):
            setattr(
                self.model,
                "_dflash_in_real_propose",
                getattr(self, "_dflash_in_real_propose", False),
            )
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        if (
            (
                os.environ.get("VLLM_DFLASH_LOGIT_DEBUG") == "1"
                or _dflash_trace_enabled()
            )
            and not getattr(self, "_logged_dflash_raw_logit_debug", False)
        ):
            self._logged_dflash_raw_logit_debug = True
            top_draft_ids = logits.argmax(dim=-1)
            top_target_ids = top_draft_ids + self.draft_id_to_target_id[top_draft_ids]
            hs = hidden_states.detach().float()
            hs_norm = hs.norm(dim=-1)
            cos_offdiag_mean = float("nan")
            cos_offdiag_max = float("nan")
            if hs.shape[0] > 1:
                hs_unit = torch.nn.functional.normalize(hs, dim=-1)
                cos = hs_unit @ hs_unit.T
                offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool,
                                         device=cos.device)]
                cos_offdiag_mean = float(offdiag.mean().item())
                cos_offdiag_max = float(offdiag.max().item())
            logger.info(
                "DFlash trace raw logits: hidden_shape=%s, "
                "hidden_norm_mean=%.6f, hidden_absmax=%.6f, "
                "hidden_cos_offdiag_mean=%.6f, hidden_cos_offdiag_max=%.6f, "
                "raw_logits_shape=%s, top_draft_ids=%s, mapped_target_ids=%s.",
                tuple(hidden_states.shape),
                float(hs_norm.mean().item()),
                float(hs.abs().max().item()),
                cos_offdiag_mean,
                cos_offdiag_max,
                tuple(logits.shape),
                top_draft_ids[:8].detach().cpu().tolist(),
                top_target_ids[:8].detach().cpu().tolist(),
            )
        logits_new = logits.new_full(
            (logits.shape[0], self.target_vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def target_ids_to_draft_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.target_id_to_draft_id is None:
            return input_ids
        clamped = input_ids.clamp(min=0, max=self.target_id_to_draft_id.shape[0] - 1)
        return self.target_id_to_draft_id[clamped]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Precompute projected + RoPE'd K/V and write to cache."""
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    @property
    def sliding_attention_layer_names(self) -> set[str]:
        return self.model.sliding_attention_layer_names

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        if os.environ.get("VLLM_DFLASH_NORM_BEFORE_FC") == "1":
            hidden_states = F.rms_norm(
                hidden_states,
                (hidden_states.shape[-1],),
                eps=self.config.rms_norm_eps,
            )
        result = self.model.fc(hidden_states)
        if (
            (
                os.environ.get("VLLM_DFLASH_AUX_DEBUG") == "1"
                or _dflash_trace_enabled()
            )
            and not getattr(self, "_logged_dflash_aux_debug", False)
        ):
            self._logged_dflash_aux_debug = True
            flat = hidden_states.detach().float()
            out = result.detach().float()
            logger.info(
                "DFlash trace aux/fc: fc_input_shape=%s, fc_output_shape=%s, "
                "fc_input_norm_mean=%.6f, fc_input_absmax=%.6f, "
                "fc_output_norm_mean=%.6f, fc_output_absmax=%.6f, "
                "fc_output_cos_offdiag_mean=%.6f.",
                tuple(hidden_states.shape),
                tuple(result.shape),
                float(flat.norm(dim=-1).mean().item()),
                float(flat.abs().max().item()),
                float(out.norm(dim=-1).mean().item()),
                float(out.abs().max().item()),
                _dflash_hidden_debug_stats(result)[2],
            )
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_draft_id_mapping = False
        includes_lm_head = False
        includes_embed_tokens = False
        for name, loaded_weight in weights:
            assert "mask_hidden" not in name, (
                "DFlash should use mask_token_id to embed the padding hidden state"
            )
            if "t2d" in name:
                continue
            elif "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif "lm_head" not in name:
                name = "model." + name
            if "lm_head" in name:
                includes_lm_head = True
                self.has_own_lm_head = True
            if "embed_tokens" in name:
                includes_embed_tokens = True
                self.has_own_embed_tokens = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        skip_substrs = []
        if not includes_draft_id_mapping:
            if self.draft_id_to_target_id is not None:
                raise ValueError(
                    "DFlash checkpoint uses a truncated draft vocab "
                    f"({self.config.draft_vocab_size} != {self.target_vocab_size}) "
                    "but does not include a d2t token mapping."
                )
            skip_substrs.append("draft_id_to_target_id")
        skip_substrs.append("target_id_to_draft_id")
        if self.draft_id_to_target_id is not None and not includes_lm_head:
            raise ValueError(
                "DFlash checkpoint uses a truncated draft vocab "
                f"({self.config.draft_vocab_size} != {self.target_vocab_size}) "
                "but does not include a draft lm_head."
            )
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=None,
            skip_substrs=skip_substrs,
        )
        loaded_params = loader.load_weights(model_weights.items())
        if os.environ.get("VLLM_DFLASH_WEIGHT_AUDIT") == "1":
            expected_params = set(dict(self.named_parameters()).keys())
            expected_buffers = set(dict(self.named_buffers()).keys())
            expected_keys = expected_params | expected_buffers
            checkpoint_keys = set(model_weights.keys())
            consumed_checkpoint_keys = set(loaded_params)
            for key in checkpoint_keys:
                remapped_key = key
                if "midlayer." in remapped_key:
                    remapped_key = remapped_key.replace("midlayer.", "layers.0.")
                for param_name, weight_name in (
                    (".qkv_proj", ".q_proj"),
                    (".qkv_proj", ".k_proj"),
                    (".qkv_proj", ".v_proj"),
                    (".gate_up_proj", ".gate_proj"),
                    (".gate_up_proj", ".up_proj"),
                ):
                    if weight_name in remapped_key:
                        remapped_key = remapped_key.replace(weight_name, param_name)
                        break
                if remapped_key in loaded_params:
                    consumed_checkpoint_keys.add(key)

            unused_checkpoint_keys = sorted(checkpoint_keys - consumed_checkpoint_keys)
            missing_model_keys = sorted(
                key
                for key in expected_keys - loaded_params
                if not any(substr in key for substr in skip_substrs)
                and not key.endswith("rotary_emb.cos_sin_cache")
                and "self_attn.attn._" not in key
            )
            logger.info(
                "DFlash weight audit: checkpoint_keys=%d loaded=%d "
                "unused_checkpoint_keys=%d missing_model_keys=%d.",
                len(checkpoint_keys),
                len(loaded_params),
                len(unused_checkpoint_keys),
                len(missing_model_keys),
            )
            if unused_checkpoint_keys:
                logger.info(
                    "DFlash weight audit unused checkpoint keys sample: %s",
                    unused_checkpoint_keys[:50],
                )
            if missing_model_keys:
                logger.info(
                    "DFlash weight audit missing model keys sample: %s",
                    missing_model_keys[:50],
                )
        if self.draft_id_to_target_id is not None and includes_draft_id_mapping:
            target_ids = (
                torch.arange(
                    self.config.draft_vocab_size,
                    device=self.draft_id_to_target_id.device,
                    dtype=self.draft_id_to_target_id.dtype,
                )
                + self.draft_id_to_target_id
            )
            draft_ids = torch.arange(
                self.config.draft_vocab_size,
                device=self.target_id_to_draft_id.device,
                dtype=self.target_id_to_draft_id.dtype,
            )
            self.target_id_to_draft_id.data.zero_()
            valid = (target_ids >= 0) & (target_ids < self.target_vocab_size)
            self.target_id_to_draft_id.data[target_ids[valid]] = draft_ids[valid]
            logger.info(
                "Loaded DFlash draft_id_to_target_id mapping for draft vocab "
                "size %d (target vocab size %d).",
                self.config.draft_vocab_size,
                self.target_vocab_size,
            )
        if os.environ.get("VLLM_DFLASH_WEIGHT_DEBUG") == "1":
            fc_weight = getattr(self.model.fc, "weight", None)
            if fc_weight is not None:
                logger.info(
                    "DFlash weight debug: fc.weight shape=%s dtype=%s "
                    "min=%.6f max=%.6f.",
                    tuple(fc_weight.shape),
                    fc_weight.dtype,
                    float(fc_weight.min().item()),
                    float(fc_weight.max().item()),
                )
            logger.info(
                "DFlash weight debug: lm_head.weight shape=%s dtype=%s "
                "min=%.6f max=%.6f.",
                tuple(self.lm_head.weight.shape),
                self.lm_head.weight.dtype,
                float(self.lm_head.weight.min().item()),
                float(self.lm_head.weight.max().item()),
            )
            if hasattr(self.model, "embed_tokens"):
                logger.info(
                    "DFlash weight debug: embed_tokens.weight shape=%s dtype=%s "
                    "min=%.6f max=%.6f.",
                    tuple(self.model.embed_tokens.weight.shape),
                    self.model.embed_tokens.weight.dtype,
                    float(self.model.embed_tokens.weight.min().item()),
                    float(self.model.embed_tokens.weight.max().item()),
                )
            if self.draft_id_to_target_id is not None:
                logger.info(
                    "DFlash weight debug: d2t shape=%s min=%d max=%d, "
                    "t2d shape=%s min=%d max=%d.",
                    tuple(self.draft_id_to_target_id.shape),
                    int(self.draft_id_to_target_id.min().item()),
                    int(self.draft_id_to_target_id.max().item()),
                    tuple(self.target_id_to_draft_id.shape),
                    int(self.target_id_to_draft_id.min().item()),
                    int(self.target_id_to_draft_id.max().item()),
                )
        self.model._build_fused_kv_buffers()
