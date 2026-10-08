# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ParallelConfig
from vllm.model_executor.models.dspark_ngram_markov import DSparkNgramMarkovAdapter
from vllm.model_executor.models.dspark_prefix_reranker import DSparkPrefixReranker
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import _get_dspark_parallel_config
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


@dataclass
class _FakeEPLBConfig:
    num_redundant_experts: int = 0


@dataclass
class _FakeParallelConfig:
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 8
    enable_eplb: bool = True
    eplb_config: _FakeEPLBConfig = field(
        default_factory=lambda: _FakeEPLBConfig(num_redundant_experts=32)
    )
    enable_elastic_ep: bool = True

    def __post_init__(self) -> None:
        if not self.enable_eplb and self.eplb_config.num_redundant_experts:
            raise ValueError("redundant experts require EPLB")
        if self.enable_elastic_ep and not self.enable_eplb:
            raise ValueError("elastic EP requires EPLB")


def test_dspark_parallel_config_disables_eplb_atomically():
    target_config = _FakeParallelConfig()

    draft_config = _get_dspark_parallel_config(
        target_config,
        tensor_parallel_size=4,
    )

    assert target_config.pipeline_parallel_size == 2
    assert target_config.tensor_parallel_size == 8
    assert target_config.enable_eplb
    assert target_config.eplb_config.num_redundant_experts == 32
    assert target_config.enable_elastic_ep

    assert draft_config is not target_config
    assert draft_config.pipeline_parallel_size == 1
    assert draft_config.tensor_parallel_size == 4
    assert not draft_config.enable_eplb
    assert draft_config.eplb_config.num_redundant_experts == 0
    assert not draft_config.enable_elastic_ep
    assert draft_config.eplb_config is not target_config.eplb_config


def test_prefix_reranker_walk_reads_selected_hidden_rows_and_request_anchors():
    torch.manual_seed(11)
    reranker = DSparkPrefixReranker(6, 4, 3, width=8, num_heads=2, top_k=3)
    with torch.no_grad():
        reranker.residual_out.weight.normal_()
    w1, w2 = torch.randn(9, 4), torch.randn(9, 4)
    lm_head = torch.randn(9, 6)
    embed = lambda ids: torch.nn.functional.embedding(ids, w1)
    bias = lambda e: torch.nn.functional.linear(e, w2)
    logits = lambda h: torch.nn.functional.linear(h, lm_head)
    speculator = DSparkSpeculator.__new__(DSparkSpeculator)
    speculator.model = SimpleNamespace(
        model=SimpleNamespace(
            prefix_reranker=reranker,
            markov_head=SimpleNamespace(markov_w2=SimpleNamespace(weight=w2)),
        ),
        compute_draft_logits=logits,
        markov_embed=embed,
        markov_bias=bias,
    )
    speculator.num_speculative_steps = 3
    speculator.sample_indices = torch.tensor([3, 1, 5, 2, 6, 0])
    speculator._anchor_idx = torch.tensor([0, 3])
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.tensor([2, 0, 0, 7, 0, 0])
    )
    speculator.draft_tokens = torch.full((3, 3), -1, dtype=torch.long)
    head_hidden = torch.randn(7, 6)
    selected = head_hidden[speculator.sample_indices].view(2, 3, 6)
    expected = reranker.sample(
        logits(selected), selected, torch.tensor([2, 7]), embed, bias, w2
    )
    speculator._sample_sequential(2, head_hidden)
    torch.testing.assert_close(speculator.draft_tokens[:2], expected)
    assert (speculator.draft_tokens[2] == -1).all()


def test_ngram_markov_walk_uses_generated_prefix_and_resets_each_round():
    torch.manual_seed(3)
    adapter = DSparkNgramMarkovAdapter(6, 4, 1e-6)
    with torch.no_grad():
        adapter.ngram_value_proj.weight.normal_()
    w1, w2, lm = torch.randn(11, 4), torch.randn(11, 4), torch.randn(11, 6)
    embed = lambda ids: torch.nn.functional.embedding(ids, w1)
    bias = lambda m: torch.nn.functional.linear(m, w2)
    logits = lambda h: torch.nn.functional.linear(h, lm)
    calls = []

    class Lookup:
        context_length = 2
        eos_token_id = 10

        def __call__(self, previous, context, dtype):
            calls.append((previous.clone(), context.clone()))
            return torch.cat([context, previous[:, None]], -1).float().repeat(1, 2)

    spec = DSparkSpeculator.__new__(DSparkSpeculator)
    spec.model = SimpleNamespace(
        model=SimpleNamespace(prefix_reranker=None),
        compute_draft_logits=logits,
        markov_embed=embed,
        ngram_markov_embed=lambda ids, h, n: adapter(embed(ids), h, n),
        markov_bias=bias,
    )
    spec.ngram_lookup = Lookup()
    spec.num_speculative_steps = 3
    spec.sample_indices = torch.tensor([3, 1, 5, 2, 6, 0])
    spec.sample_idx_mapping = torch.tensor([1, 1, 1, 0, 0, 0])
    spec.sample_pos = torch.zeros(6, dtype=torch.long)
    spec._anchor_idx = torch.tensor([0, 3])
    spec.input_buffers = SimpleNamespace(
        input_ids=torch.tensor([2, 0, 0, 7, 0, 0]),
        positions=torch.tensor([3, 4, 5, 1, 2, 3]),
    )
    spec.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[1, 8, 9, 9], [3, 4, 5, 9]]))
    )
    spec.draft_tokens = torch.full((3, 3), -1, dtype=torch.long)
    spec._draft_topk = None
    spec.use_confidence_head = False
    spec._sample_logits = lambda scores, *args: scores.argmax(-1)
    head_hidden = torch.randn(7, 6)
    spec._sample_sequential(2, head_hidden)
    selected = head_hidden[spec.sample_indices].view(2, 3, 6)
    previous, context = torch.tensor([2, 7]), torch.tensor([[4, 5], [10, 1]])
    expected = []
    for i in range(3):
        torch.testing.assert_close(calls[i][0], previous)
        torch.testing.assert_close(calls[i][1], context)
        raw = torch.cat([context, previous[:, None]], -1).float().repeat(1, 2)
        m = adapter(embed(previous), selected[:, i], raw)
        context = torch.cat([context[:, 1:], previous[:, None]], -1)
        previous = (logits(selected[:, i]) + bias(m)).argmax(-1)
        expected.append(previous)
    torch.testing.assert_close(spec.draft_tokens[:2], torch.stack(expected, -1))
    assert (spec.draft_tokens[2] == -1).all()
    first_context = calls[0][1].clone()
    calls.clear()
    spec._sample_sequential(2, head_hidden)
    torch.testing.assert_close(calls[0][1], first_context)


@pytest.mark.parametrize("pcp_size", [1, 4])
@pytest.mark.parametrize("dcp_size", [1, 4])
@pytest.mark.parametrize("use_mla", [False, True])
def test_draft_context_parallelism_without_changing_target(
    monkeypatch, pcp_size, dcp_size, use_mla
):
    target_parallel = ParallelConfig(
        tensor_parallel_size=4,
        prefill_context_parallel_size=pcp_size,
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=16,
        distributed_executor_backend="mp",
    )
    target_config = SimpleNamespace(
        parallel_config=target_parallel,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(use_mla=use_mla)
        ),
    )

    class CapturedConfig(Exception):
        pass

    def capture_init(self, config, device):
        raise CapturedConfig(config)

    monkeypatch.setattr(DraftModelSpeculator, "__init__", capture_init)
    with pytest.raises(CapturedConfig) as captured:
        DFlashSpeculator(target_config, device=None)
    draft_parallel = captured.value.args[0].parallel_config
    assert draft_parallel.tensor_parallel_size == 4
    assert draft_parallel.prefill_context_parallel_size == 1
    assert draft_parallel.decode_context_parallel_size == (dcp_size if use_mla else 1)
    assert draft_parallel.cp_kv_cache_interleave_size == 16
    assert target_parallel.prefill_context_parallel_size == pcp_size
    assert target_parallel.decode_context_parallel_size == dcp_size
    assert target_parallel.cp_kv_cache_interleave_size == 16


@pytest.mark.parametrize("configured_dcp", [1, 4])
def test_attention_uses_draft_dcp_setting_inside_target_process_group(
    monkeypatch, configured_dcp
):
    import vllm.config as config_module
    from vllm.distributed import parallel_state
    from vllm.v1.attention.backend import AttentionImplBase

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=configured_dcp)
    )
    monkeypatch.setattr(
        config_module, "get_current_vllm_config_or_none", lambda: config
    )

    monkeypatch.setattr(
        parallel_state, "_DCP", SimpleNamespace(world_size=4, rank_in_group=2)
    )
    impl = AttentionImplBase()
    assert impl.dcp_world_size == configured_dcp
    assert impl.dcp_rank == (0 if configured_dcp == 1 else 2)


@pytest.mark.parametrize("draft_dcp_size", [1, 4])
def test_runner_marks_only_replicated_draft_caches(monkeypatch, draft_dcp_size):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64, dtype=torch.bfloat16
    )
    monkeypatch.setattr(
        model_runner, "get_kv_cache_spec", lambda _: {"target": spec, "draft": spec}
    )
    runner = object.__new__(model_runner.GPUModelRunner)
    runner.vllm_config = None
    runner.dcp_size = 4
    runner.speculator = object.__new__(DFlashSpeculator)
    runner.speculator.dcp_size = draft_dcp_size
    runner.speculator.draft_attn_layer_names = {"draft"}

    specs = runner.get_kv_cache_spec()

    assert specs["target"].dcp_sharded
    assert specs["draft"].dcp_sharded == (draft_dcp_size == 4)
