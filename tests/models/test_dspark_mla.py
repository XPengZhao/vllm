# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.registry import ModelRegistry
from vllm.models.deepseek_v4.nvidia import dspark as dsv4_dspark
from vllm.models.kimi_k3.nvidia import dspark_mla
from vllm.models.kimi_k3.nvidia.dspark_mla import K3DSparkForCausalLM, K3DSparkModel


def test_dspark_mla_uses_compile_free_model_entrypoint():
    assert ModelRegistry._try_load_model_cls("K3DSparkModel") is K3DSparkForCausalLM
    assert not issubclass(K3DSparkModel, TorchCompileWithNoGuardsWrapper)


@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_name", "shard_id"),
    [
        (
            "layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            0,
        ),
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "model.layers.0.self_attn.fused_qkv_a_proj.weight",
            1,
        ),
        (
            "layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        ("context_proj.weight", "model.context_proj.weight", None),
    ],
)
def test_dspark_mla_checkpoint_weight_mapping(checkpoint_name, runtime_name, shard_id):
    assert K3DSparkForCausalLM.hf_to_vllm_mapper._map_name_with_shard(
        checkpoint_name
    ) == (runtime_name, shard_id)


def test_dspark_mla_shares_frozen_target_weights_and_skips_training_head():
    assert not K3DSparkForCausalLM.has_own_embed_tokens
    assert not K3DSparkForCausalLM.has_own_lm_head
    mapper = K3DSparkForCausalLM.hf_to_vllm_mapper
    for name in ("confidence_head.weight", "embed_tokens.weight", "lm_head.weight"):
        assert mapper._map_name(name) is None


@pytest.mark.cpu_test
def test_dspark_markov_head_is_replicated(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers import logits_processor, vocab_parallel_embedding

    monkeypatch.setattr(
        vocab_parallel_embedding, "get_tensor_model_parallel_rank", lambda: 3
    )
    monkeypatch.setattr(
        vocab_parallel_embedding,
        "get_tensor_model_parallel_world_size",
        lambda: 8,
    )
    monkeypatch.setattr(
        logits_processor,
        "get_current_vllm_config",
        lambda: SimpleNamespace(model_config=None),
    )

    head = DSparkMarkovHead(128, 128, 8, prefix="markov_head")
    assert head.markov_w2.tp_size == 1
    assert head.markov_w1.weight.shape == (128, 8)
    assert head.markov_w2.weight.shape == (128, 8)
    head.markov_w2.quant_method.process_weights_after_loading(head.markov_w2)

    def fail_collective(*args, **kwargs):
        raise AssertionError("replicated Markov head must not invoke TP collectives")

    monkeypatch.setattr(
        vocab_parallel_embedding,
        "tensor_model_parallel_all_reduce",
        fail_collective,
    )
    logits_processor = LogitsProcessor(128)
    monkeypatch.setattr(logits_processor, "_gather_logits", fail_collective)

    markov_embed = head.embed(torch.tensor([1, 2]))
    bias = head.bias(markov_embed, logits_processor)
    assert markov_embed.shape == (2, 8)
    assert bias.shape == (2, 128)


@pytest.mark.cpu_test
def test_k3_dspark_uses_replicated_markov_head(monkeypatch: pytest.MonkeyPatch):
    markov_head_calls = []
    context_kv_proj_calls = []

    class DummyModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    def make_markov_head(*args, **kwargs):
        markov_head_calls.append((args, kwargs))
        return DummyModule()

    def make_context_kv_proj(*args, **kwargs):
        context_kv_proj_calls.append((args, kwargs))
        return DummyModule()

    monkeypatch.setattr(dspark_mla, "get_draft_quant_config", lambda _: None)
    monkeypatch.setattr(dspark_mla, "ReplicatedLinear", DummyModule)
    monkeypatch.setattr(dspark_mla, "MergedColumnParallelLinear", make_context_kv_proj)
    monkeypatch.setattr(dspark_mla, "RMSNorm", DummyModule)
    monkeypatch.setattr(dspark_mla, "K3DSparkDecoderLayer", DummyModule)
    monkeypatch.setattr(dspark_mla, "DSparkMarkovHead", make_markov_head)

    config = SimpleNamespace(
        target_hidden_size=16,
        num_target_layers=2,
        hidden_size=8,
        kv_lora_rank=3,
        qk_rope_head_dim=1,
        rms_norm_eps=1e-6,
        num_hidden_layers=1,
        vocab_size=128,
        draft_vocab_size=128,
        markov_rank=4,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=config)
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
    )

    K3DSparkModel(vllm_config=vllm_config, start_layer_id=0, prefix="model")

    assert len(markov_head_calls) == 1
    assert context_kv_proj_calls == [
        (
            (8, [4]),
            {
                "bias": False,
                "return_bias": False,
                "quant_config": None,
                "prefix": "model.layers.0.self_attn.fused_qkv_a_proj",
                "disable_tp": True,
            },
        )
    ]


def test_context_kv_weights_are_loaded_as_merged_linear_shards():
    weights = [
        (
            "layers.0.self_attn.kv_a_proj_with_mqa.weight_packed",
            torch.arange(4),
        ),
        (
            "layers.1.self_attn.kv_a_proj_with_mqa.weight_scale",
            torch.tensor(0.5),
        ),
    ]

    duplicated = dspark_mla._duplicate_context_kv_weights(weights, 2)
    mapped = list(K3DSparkForCausalLM.hf_to_vllm_mapper.apply(duplicated))

    assert [name for name, _ in mapped] == [
        "model.layers.0.self_attn.fused_qkv_a_proj.weight_packed",
        "model.context_kv_proj.weight_packed",
        "model.layers.1.self_attn.fused_qkv_a_proj.weight_scale",
        "model.context_kv_proj.weight_scale",
    ]
    assert [weight.shard_id for _, weight in mapped] == [1, 0, 1, 1]
    assert mapped[0][1].data_ptr() == mapped[1][1].data_ptr()
    assert mapped[2][1].data_ptr() == mapped[3][1].data_ptr()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "scale_dtype", [torch.uint8, torch.float8_e8m0fnu, torch.float32]
)
@pytest.mark.parametrize(
    ("checkpoint_name", "runtime_module", "shard_id"),
    [
        ("mtp.0.attn.wq_a.scale", "model.layers.0.attn.fused_wqa_wkv", 0),
        ("mtp.0.attn.wkv.scale", "model.layers.0.attn.fused_wqa_wkv", 1),
        ("mtp.0.main_proj.scale", "model.main_proj", None),
        (
            "mtp.0.ffn.shared_experts.w1.scale",
            "model.layers.0.ffn.shared_experts.gate_up_proj",
            0,
        ),
        (
            "mtp.0.ffn.shared_experts.w2.scale",
            "model.layers.0.ffn.shared_experts.down_proj",
            None,
        ),
    ],
)
def test_v41_dspark_loads_linear_scales(
    monkeypatch, scale_dtype, checkpoint_name, runtime_module, shard_id
):
    """Checkpoint ``.scale`` maps to the quant method's scale parameter and
    loads untouched. MXFP8 block-scale expansion lives in the KMxfp8Static
    loader (see tests/quantization/test_modelopt.py), not in load_weights."""
    from vllm.models.deepseek_v41.nvidia import dspark

    mxfp8 = scale_dtype != torch.float32
    scale_name = "weight_scale" if mxfp8 else "weight_scale_inv"
    runtime_name = f"{runtime_module}.{scale_name}"
    raw = torch.tensor([[120, 127], [128, 130]], dtype=torch.uint8)
    checkpoint_scale = raw.view(scale_dtype) if mxfp8 else raw.float()
    param = nn.Parameter(torch.empty_like(checkpoint_scale), requires_grad=False)
    shards = []

    def load_scale(param, weight, *args):
        shards.append(args)
        assert weight.dtype == checkpoint_scale.dtype
        param.copy_(weight)

    param.weight_loader = load_scale
    draft = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=4, n_routed_experts=1),
        quant_config=SimpleNamespace(
            weight_block_size=[32, 32] if mxfp8 else [128, 128]
        ),
        linear_scale_name=scale_name,
        pad_shared_expert=False,
        model=SimpleNamespace(
            layers=[SimpleNamespace(ffn=SimpleNamespace(use_native_mega_moe=False))],
            confidence_head=None,
        ),
        named_parameters=lambda: [(runtime_name, param)],
        process_weights_after_loading=lambda: None,
    )
    draft._remap_dspark_name = lambda name: (
        dspark.DSparkDeepseekV4ForCausalLM._remap_dspark_name(draft, name)
    )
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(dspark, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        dspark, "fused_moe_make_expert_params_mapping", lambda *a, **kw: []
    )

    loaded = dspark.DSparkDeepseekV4ForCausalLM.load_weights(
        draft, [(checkpoint_name, checkpoint_scale)]
    )

    assert loaded == {runtime_name}
    assert shards == [() if shard_id is None else (shard_id,)]
    torch.testing.assert_close(param, checkpoint_scale)


def test_dsv4_context_wkv_weights_are_duplicated_by_draft_layer():
    weights = [
        ("mtp.0.attn.wkv.weight", torch.arange(4)),
        ("mtp.1.attn.wq_a.weight", torch.arange(3)),
        ("mtp.2.attn.wkv.scale", torch.tensor(0.5)),
        ("mtp.3.attn.wkv.weight", torch.arange(2)),
    ]

    duplicated = list(dsv4_dspark._duplicate_context_wkv_weights(weights, 3))

    assert [name for name, _ in duplicated] == [
        "mtp.0.attn.wkv.weight",
        "context_wkv_proj.weight",
        "mtp.1.attn.wq_a.weight",
        "mtp.2.attn.wkv.scale",
        "context_wkv_proj.scale",
        "mtp.3.attn.wkv.weight",
    ]
    assert duplicated[1][1].shard_id == 0
    assert duplicated[4][1].shard_id == 2
    assert duplicated[0][1].data_ptr() == duplicated[1][1].data_ptr()
    assert duplicated[3][1].data_ptr() == duplicated[4][1].data_ptr()


def test_dsv4_context_kv_uses_one_stacked_wkv_projection(monkeypatch):
    calls = []
    stacked_output = torch.arange(24, dtype=torch.float32).view(2, 12)

    class StackedProjection:
        def __init__(self):
            self.calls = 0

        def __call__(self, main_x):
            self.calls += 1
            assert main_x.shape == (2, 5)
            return stacked_output

    projection = StackedProjection()
    layers = [
        SimpleNamespace(attn=SimpleNamespace(kv_norm=lambda kv, offset=i: kv + offset))
        for i in range(3)
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(head_dim=4),
        context_wkv_proj=projection,
        layers=layers,
        num_dspark_layers=3,
    )
    slot_mappings = [torch.tensor([0, 1]), None, torch.tensor([4, 5])]
    monkeypatch.setattr(
        dsv4_dspark,
        "_insert_context_kv",
        lambda attn, kv, positions, slots: calls.append(
            (attn, kv.clone(), positions, slots)
        ),
    )

    dsv4_dspark.DSparkDeepseekV4Model.precompute_and_store_context_kv(
        model,
        torch.zeros(2, 5),
        torch.tensor([7, 8]),
        slot_mappings,
    )

    assert projection.calls == 1
    assert len(calls) == 2
    assert torch.equal(calls[0][1], stacked_output.view(2, 3, 4)[:, 0])
    assert torch.equal(calls[1][1], stacked_output.view(2, 3, 4)[:, 2] + 2)
    assert calls[0][3] is slot_mappings[0]
    assert calls[1][3] is slot_mappings[2]


@pytest.mark.cpu_test
def test_k3_dspark_mla_kv_cache_spec_groups_with_target_mla():
    """The draft's MLA layers must share a KV cache group with the target's."""
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
    from vllm.v1.core.kv_cache_utils import _get_kv_cache_groups_uniform_page_size

    vllm_config = SimpleNamespace(
        model_config=None, cache_config=SimpleNamespace(block_size=64)
    )
    target_attn = SimpleNamespace(
        kv_cache_dtype="fp8",
        head_size=576,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
        attn_backend=SimpleNamespace(get_name=lambda: "ROCM_AITER_MLA"),
        _uses_flat_kv_cache=lambda: False,
    )
    draft_attn = SimpleNamespace(
        kv_cache_dtype="fp8", head_size=576, non_causal_multi_token_decode=True
    )
    target_spec = MLAAttention.get_kv_cache_spec(target_attn, vllm_config)
    draft_spec = MultiHeadLatentAttention.get_kv_cache_spec(draft_attn, vllm_config)

    kv_cache_spec = {f"model.layers.{i}.attn": target_spec for i in range(24)}
    kv_cache_spec |= {f"draft.layers.{i}.attn": draft_spec for i in range(5)}

    groups = _get_kv_cache_groups_uniform_page_size(kv_cache_spec)
    assert len(groups) == 1
    assert len(groups[0].layer_names) == 29


@pytest.fixture
def qwen_mla_module():
    from vllm.model_executor.models import qwen3_dspark_mla

    return qwen3_dspark_mla


def _small_qwen_mla(module, query_width):
    """Exercise projection math independently of distributed initialization."""

    class Rotary(nn.Module):
        def forward(self, positions, query, key):
            shape = query.shape
            query = query.view(len(positions), -1, 2)
            angle = positions.float()[:, None, None]
            rotated = torch.stack((-query[..., 1], query[..., 0]), dim=-1)
            return (query * angle.cos() + rotated * angle.sin()).view(shape), key

    torch.manual_seed(42)
    attention = module.Qwen3DSparkMLA.__new__(module.Qwen3DSparkMLA)
    nn.Module.__init__(attention)
    attention.num_heads = 2
    attention.rank = 3
    attention.nope_dim = attention.rope_dim = 2
    attention.head_dim = attention.v_dim = 4
    attention.q_proj = nn.Linear(query_width, 8, bias=False)
    attention.kv_a_proj = nn.Linear(6, 5, bias=False)
    attention.kv_a_norm = nn.RMSNorm(3, eps=1e-6)
    attention.kv_a_norm.weight.data.copy_(torch.tensor([0.7, 1.0, 1.3]))
    attention.kv_b_proj = nn.Linear(3, 12, bias=False)
    attention.o_proj = nn.Linear(8, query_width, bias=False)
    attention.rotary_emb = Rotary()
    return attention


@pytest.mark.parametrize("query_width", [3, 6])
def test_qwen_mla_projects_shared_rope_and_normalized_latent(
    qwen_mla_module, query_width
):
    """Backbone/head must match the training formula, including latent norm."""
    attention = _small_qwen_mla(qwen_mla_module, query_width)
    context = torch.randn(4, 6)
    hidden = torch.randn(4, query_width)
    positions = torch.tensor([0, 3, 7, 11])
    query = attention.project_query(positions, hidden)
    key, value = attention.project_kv(positions, context)

    def rotate(x):
        angle = positions.float()[:, None]
        while angle.ndim < x.ndim:
            angle = angle.unsqueeze(-1)
        return x * angle.cos() + torch.stack((-x[..., 1], x[..., 0]), -1) * angle.sin()

    raw_q = torch.nn.functional.linear(hidden, attention.q_proj.weight).view(4, 2, 4)
    raw_kv = torch.nn.functional.linear(context, attention.kv_a_proj.weight)
    latent = raw_kv[:, :3]
    latent = latent * torch.rsqrt(latent.square().mean(-1, keepdim=True) + 1e-6)
    latent = latent * attention.kv_a_norm.weight
    expanded = torch.nn.functional.linear(latent, attention.kv_b_proj.weight)
    expanded = expanded.view(4, 2, 6)
    expected_q = torch.cat((raw_q[..., :2], rotate(raw_q[..., 2:])), -1)
    expected_k = torch.cat(
        (expanded[..., :2], rotate(raw_kv[:, 3:])[:, None].expand(-1, 2, -1)), -1
    )
    torch.testing.assert_close(query, expected_q)
    torch.testing.assert_close(key, expected_k)
    torch.testing.assert_close(value, expanded[..., 2:])
    # Chunking the accepted context cannot change its cached projection.
    chunks = [
        attention.project_kv(positions[s], context[s])
        for s in (slice(0, 2), slice(2, 4))
    ]
    torch.testing.assert_close(key, torch.cat([chunk[0] for chunk in chunks]))
    torch.testing.assert_close(value, torch.cat([chunk[1] for chunk in chunks]))


def test_qwen_markov_mla_reads_cache_without_residual(qwen_mla_module):
    attention = _small_qwen_mla(qwen_mla_module, 3)
    embedding = torch.randn(2, 3)
    positions = torch.tensor([4, 9])
    attention.markov_w2 = nn.Linear(3, 16, bias=False)
    cached_output = torch.randn(2, 8)

    class CachedAttention(nn.Module):
        def forward(self, query, key, value):
            assert key is None and value is None
            torch.testing.assert_close(
                query, attention.project_query(positions, embedding).flatten(1)
            )
            return cached_output

    attention.attn = CachedAttention()
    actual = qwen_mla_module.Qwen3MarkovMLA.bias(
        attention, embedding, positions, lambda head, x: head(x)
    )
    expected = attention.markov_w2(attention.o_proj(cached_output))
    torch.testing.assert_close(actual, expected)


def test_qwen_markov_metadata_excludes_anchor_and_padded_requests(qwen_mla_module):
    metadata = qwen_mla_module.FlashAttentionMetadata(
        num_actual_tokens=6,
        max_query_len=3,
        query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
        max_seq_len=20,
        seq_lens=torch.tensor([10, 20], dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        slot_mapping=torch.arange(6),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
    )
    result = qwen_mla_module.markov_attention_metadata(
        metadata, 2, torch.arange(3, dtype=torch.int32), torch.tensor([7, 0])
    )
    assert result.seq_lens.tolist() == [7, 0]
    assert result.query_start_loc.tolist() == [0, 1, 2]
    assert result.max_query_len == 1 and result.num_actual_tokens == 2
    assert not result.causal
    assert result.scheduler_metadata is None
    torch.testing.assert_close(result.block_table, metadata.block_table)
    assert metadata.seq_lens.tolist() == [10, 20]


@pytest.fixture
def qwen_speculator_class():
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    return DSparkSpeculator


def test_qwen_markov_sampling_uses_previous_token_and_raw_confidence_embedding(
    qwen_speculator_class,
):
    speculator = qwen_speculator_class.__new__(qwen_speculator_class)
    speculator._draft_topk = None
    speculator.num_speculative_steps = 2
    speculator.max_model_len = 32
    speculator.sample_indices = torch.arange(4)
    speculator.sample_idx_mapping = torch.tensor([0, 0, -1, -1])
    speculator.sample_pos = torch.tensor([6, 7, 0, 0])
    speculator.input_buffers = SimpleNamespace(input_ids=torch.tensor([3, 0, 0, 0]))
    speculator._anchor_idx = torch.tensor([0, 2])
    speculator.use_confidence_head = True
    speculator.draft_tokens = torch.zeros(2, 2, dtype=torch.long)
    speculator.draft_token_confidence_probs = torch.zeros(2, 2)
    calls = []
    confidence_inputs = []

    def bias(embedding, positions):
        calls.append((embedding.clone(), positions.clone()))
        logits = torch.zeros(2, 8)
        logits[:, len(calls)] = 5
        return logits

    def confidence(hidden, embedding):
        confidence_inputs.append(embedding.clone())
        return torch.ones(4)

    speculator.model = SimpleNamespace(
        has_markov_mla=True,
        compute_draft_logits=lambda hidden: torch.zeros(4, 8),
        markov_embed=lambda ids: ids.float()[:, None].expand(-1, 3),
        markov_bias=bias,
        compute_confidence=confidence,
    )
    speculator._sample_logits = lambda logits, *args: logits.argmax(-1)
    speculator._sample_sequential(2, torch.zeros(4, 6))
    assert speculator.draft_tokens.tolist() == [[1, 2], [0, 0]]
    assert calls[0][1].tolist() == [5, 0]
    assert calls[1][1].tolist() == [6, 0]
    assert calls[0][0][:, 0].tolist() == [3, 0]
    assert calls[1][0][:, 0].tolist() == [1, 0]
    assert confidence_inputs[0][:, 0].tolist() == [3, 1, 0, 0]
