# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for actual runtime methods with simulated stream/event plumbing."""

import ast
import importlib.util
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm/v1/worker/gpu/spec_decode/dspark/late_speculator.py"
)


def runtime_class():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LateDSparkSpeculator"
    )
    stage_spec = importlib.util.spec_from_file_location(
        "late_stages", SOURCE.parents[5] / "model_executor/models/dspark_late.py"
    )
    assert stage_spec is not None and stage_spec.loader is not None
    module = importlib.util.module_from_spec(stage_spec)
    stage_spec.loader.exec_module(module)

    scope = dict(
        device_ints=module.device_ints,
        torch=torch,
        Callable=Callable,
        DraftModelSpeculator=object,
        DSparkSpeculator=SimpleNamespace(_sample_logits=lambda *a: None),
        RequestContext=lambda slot: SimpleNamespace(
            slot=slot, length=0, bank=None, bank_start=0, ready=None
        ),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["LateDSparkSpeculator"]


def test_configuration_accepts_both_bank_versions_and_rejects_unsafe_modes():
    tree = ast.parse(SOURCE.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "validate_late_config"
    )
    scope: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), scope)
    validate = scope["validate_late_config"]
    hf = SimpleNamespace(
        num_hidden_layers=5,
        block_size=7,
        attention_bias=False,
        hidden_act="silu",
        layer_types=["full_attention"] * 5,
        rope_parameters={"rope_type": "default"},
        dflash_config={"late_conditioning": {"bank_size": 13, "max_context_lag": 8}},
    )
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf),
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1),
            num_speculative_tokens=7,
            enable_adaptive_verification=False,
            num_speculative_tokens_per_batch_size=None,
            quantization=None,
            dspark_draft_topk=None,
        ),
        parallel_config=SimpleNamespace(),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="qwen3"),
            quantization=None,
            supports_multimodal_inputs=False,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        lora_config=None,
    )
    validate(config)
    hf.dflash_config["late_conditioning"] = {"enabled": True}
    validate(config)
    hf.dflash_config["late_conditioning"].update(bank_size=12, max_context_lag=7)
    validate(config)
    hf.dflash_config["late_conditioning"]["max_context_lag"] = 8
    with pytest.raises(ValueError, match="bank12/lag7 or bank13/lag8"):
        validate(config)
    hf.dflash_config["late_conditioning"]["bank_size"] = 13
    config.scheduler_config.async_scheduling = True
    with pytest.raises(ValueError, match="no-async"):
        validate(config)
    config.scheduler_config.async_scheduling = False
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="prefix-caching"):
        validate(config)


@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("accepted", range(8))
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("cursor_upper_bound", [0, 7])
def test_prefill_mixed_acceptance_reorder_and_request_reuse(
    monkeypatch, bank_size, accepted, overlap, cursor_upper_bound
):
    helper_spec = importlib.util.spec_from_file_location(
        "stage_tests", Path(__file__).with_name("test_dspark_late.py")
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    ref, stages = helper.pair(bank_size=bank_size)
    actions = []
    prepared = []
    prepare_bank = stages.prepare_bank_batch

    def prepare(slots, lengths, starts):
        prepared.append(list(zip(slots, lengths, starts)))
        return prepare_bank(slots, lengths, starts)

    stages.prepare_bank_batch = prepare

    class Stream:
        def wait_event(self, event):
            actions.append(("wait", event))

    class Event:
        def record(self, stream):
            actions.append(("record", self))

    main = Stream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: main)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda *a: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *a: None)
    cls = runtime_class()
    runtime = cls.__new__(cls)
    runtime.contexts = {}
    runtime.pending_jobs = []
    runtime.device = "cpu"
    runtime.stream = Stream()
    runtime.overlap = overlap
    runtime.stages = stages
    runtime.draft_tokens = torch.empty(2, 7, dtype=torch.long)
    runtime._copy_request_inputs = lambda *a: None

    sampling_calls = []

    def sample_logits(logits, mapping, position, step):
        sampling_calls.append((mapping.numel(), step))
        # The GPU sampler requires one 2-D vocabulary row per request.
        assert logits.shape == (mapping.numel(), 32)
        assert mapping.shape == position.shape
        return logits.argmax(-1)

    runtime._sample_logits = sample_logits
    embedding = nn.Embedding(32, 16)
    lm_head = nn.Linear(16, 32, bias=False)
    stages.model.model.embed_tokens = embedding
    runtime.model = SimpleNamespace(
        model=stages.model.model,
        compute_draft_logits=lm_head,
        markov_embed=ref.markov_head.get_prev_embeddings,
        markov_bias=ref.markov_head.project_bias,
    )

    def batch(ids, slots, starts, widths, drafts):
        return SimpleNamespace(
            req_ids=ids,
            num_reqs=len(ids),
            idx_mapping=torch.tensor(slots),
            idx_mapping_np=np.array(slots),
            num_computed_tokens_np=np.array(starts)
            + np.where(np.array(drafts) > 0, cursor_upper_bound, 0),
            query_start_loc_np=np.cumsum([0] + widths),
            query_start_loc=torch.tensor(np.cumsum([0] + widths)),
            positions=torch.cat(
                [
                    torch.arange(start, start + width)
                    for start, width in zip(starts, widths)
                ]
            ),
            num_draft_tokens_per_req=np.array(drafts),
        )

    def propose(b, features, sampled, rejected):
        return runtime.propose(
            b,
            None,
            None,
            None,
            [features[..., :16], features[..., 16:]],
            torch.tensor(sampled),
            torch.tensor(rejected),
            # V2 RequestState.last_sampled_tokens is [max_num_reqs, 1].
            torch.tensor([[3], [4]]),
            None,
            torch.ones(2),
            torch.zeros(2, dtype=torch.long),
        ).clone()

    first = batch(["a", "b"], [1, 0], [0, 0], [5, 7], [0, 0])
    context_a, context_b = torch.randn(5, 32), torch.randn(7, 32)
    propose(first, torch.cat((context_a, context_b)), [1, 1], [0, 0])
    second = batch(["b", "a"], [0, 1], [7, 5], [8, 8], [7, 7])
    runtime.begin_verification(second)
    prefix_event = runtime.prefix_ready
    assert all(s.bank is None for s in runtime.contexts.values())
    actions.append(("target_enqueued", None))
    runtime.launch_verification_bank()
    # The draft stream waits on the event recorded BEFORE target enqueue;
    # waiting on the current main stream here would serialize verification.
    assert actions[0] == ("record", prefix_event)
    assert actions[1][0] == "target_enqueued"
    assert actions[2] == ("wait", prefix_event)
    bank_a = runtime.contexts["a"].bank.clone()
    bank_b = runtime.contexts["b"].bank.clone()
    fresh_b, fresh_a = torch.randn(8, 32), torch.randn(8, 32)
    actual = propose(
        second, torch.cat((fresh_b, fresh_a)), [accepted + 1, 1], [7 - accepted, 7]
    )
    assert sampling_calls == [(2, step) for step in range(7)] * 2
    assert runtime.contexts["b"].length == 8 + accepted
    assert runtime.contexts["a"].length == 6  # none accepted
    fallback = bank_size == 12 and accepted == 7
    assert len(prepared) == 2 + int(fallback)
    assert len(prepared[0]) == len(prepared[1]) == 2
    if fallback:
        assert prepared[-1] == [(0, 15, 16)]
    expected = []
    for slot, length, bank, bank_start, bonus, context in (
        (
            0,
            8 + accepted,
            bank_b,
            9,
            3,
            torch.cat((context_b, fresh_b[: accepted + 1])),
        ),
        (1, 6, bank_a, 7, 4, torch.cat((context_a, fresh_a[:1]))),
    ):
        if bank_size == 12:
            old = length if slot == 0 and fallback else bank_start - 2
            _, h = helper.legacy_training_forward(
                ref,
                context.unsqueeze(0),
                embedding(torch.tensor([[bonus]])),
                length,
                old,
            )
        else:
            h = stages.finish(
                slot, length, bank, bank_start, embedding(torch.tensor([bonus]))
            )
        tokens, _ = ref.markov_head.sample_block_tokens(
            lm_head(h).unsqueeze(0),
            first_prev_token_ids=torch.tensor([bonus]),
            hidden_states=h.unsqueeze(0),
            temperature=0.0,
        )
        expected.append(tokens[0])
    torch.testing.assert_close(actual, torch.stack(expected))
    assert actual.shape == (2, 7)
    assert all(
        state.bank is None and state.ready is None
        for state in runtime.contexts.values()
    )
    # A preempted/reused request starts with no stale bank or old cursor.
    runtime.remove_request("a")
    assert "a" not in runtime.contexts
    reused = batch(["a"], [1], [0], [3], [0])
    propose(reused, torch.randn(3, 32), [1], [0])
    assert runtime.contexts["a"].length == 3
    invalid = batch(["a"], [1], [4], [8], [7])
    with pytest.raises(ValueError, match="cursor"):
        propose(invalid, torch.randn(8, 32), [1], [7])
    assert runtime.contexts["a"].length == 3


def test_chunked_prefill_does_not_prepare_a_delayed_bank(monkeypatch):
    cls = runtime_class()
    runtime = cls.__new__(cls)
    runtime.contexts = {"a": SimpleNamespace(length=1024, bank=None)}
    runtime.device = "cpu"
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: None)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda *a: None)
    )
    b = SimpleNamespace(
        req_ids=["a"],
        num_computed_tokens_np=[1024],
        num_draft_tokens_per_req=np.array([0]),
    )
    runtime.begin_verification(b)
    assert not runtime.pending_jobs
    b.num_computed_tokens_np = [0]
    runtime.begin_verification(b)
    assert not runtime.pending_jobs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_runtime_bank_has_measured_gpu_overlap_with_verifier_work():
    """Measure device intervals, not just the presence of a second stream."""
    helper_spec = importlib.util.spec_from_file_location(
        "stage_gpu_tests", Path(__file__).with_name("test_dspark_late.py")
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    _, stages = helper.pair(torch.bfloat16, "cuda")
    features = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    stages.append_context(0, 0, features)
    expected = stages.prepare_bank(0, 5, 7)
    cls = runtime_class()
    runtime = cls.__new__(cls)
    runtime.contexts = {"a": SimpleNamespace(slot=0, length=5, bank=None, ready=None)}
    runtime.device = torch.device("cuda")
    runtime.stream = torch.cuda.Stream()
    runtime.overlap = True
    runtime.stages = stages
    origin, target_start, target_end, bank_start, bank_end = (
        torch.cuda.Event(enable_timing=True) for _ in range(5)
    )
    prepare = stages.prepare_bank_batch

    def timed_bank(*args):
        bank_start.record()
        bank = prepare(*args)
        bank_end.record()
        return bank

    stages.prepare_bank_batch = timed_bank
    origin.record()
    runtime.begin_verification(
        SimpleNamespace(
            req_ids=["a"], num_computed_tokens_np=[5], num_draft_tokens_per_req=[7]
        )
    )
    target_start.record()
    # Leave resources for draft kernels; engine e2e separately checks real target
    # semantics. This models an asynchronous verifier interval on the main stream.
    torch.cuda._sleep(50_000_000)
    target_end.record()
    runtime.launch_verification_bank()
    torch.cuda.current_stream().wait_event(runtime.contexts["a"].ready)
    torch.accelerator.synchronize()
    overlap_ms = min(
        origin.elapsed_time(target_end), origin.elapsed_time(bank_end)
    ) - max(origin.elapsed_time(target_start), origin.elapsed_time(bank_start))
    assert overlap_ms > 0, f"no measured device overlap: {overlap_ms}ms"
    torch.testing.assert_close(runtime.contexts["a"].bank, expected)
