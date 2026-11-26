# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostics must preserve sampler data and map reordered persistent slots."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "ngram_trace_standalone", ROOT / "vllm/v1/worker/gpu/spec_decode/ngram_trace.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_NGRAM_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_NGRAM_TRACE_PREFIX", "ngram-test-")
    sampler = NS(
        req_states=NS(
            prefill_len=NS(np=np.array([3, 3, 3])),
            all_token_ids=NS(gpu=torch.arange(60).reshape(3, 20) % 8),
        ),
        sampling_states=NS(temperature=NS(np=np.zeros(3))),
    )
    cfg = NS(
        method="dspark",
        enable_adaptive_verification=False,
        rejection_sample_method="standard",
        draft_sample_method="greedy",
    )
    trace = module.NgramTrace(sampler, cfg)
    draft = torch.arange(48.0).reshape(3, 2, 8)
    trace.set_draft_buffers(
        NS(
            ngram_trace_logits=draft,
            ngram_trace_stamps=torch.tensor([[6, 7], [-1, -1], [11, 12]]),
        )
    )
    batch = NS(
        req_ids=["chatcmpl-ngram-test-A", "chatcmpl-ngram-test-B"],
        idx_mapping_np=np.array([2, 0]),
    )
    return trace, batch, draft


def test_reordered_slots_history_reachability_and_no_mutation(tmp_path, monkeypatch):
    trace, batch, draft = setup_trace(tmp_path, monkeypatch)
    raw = torch.arange(48.0).reshape(6, 8)
    tokens = torch.tensor([6, 7, 1, 3, 4, 5])
    positions = torch.tensor([10, 11, 12, 5, 6, 7])
    before = [x.clone() for x in (raw, draft, tokens)]
    rng = torch.random.get_rng_state()
    trace.record(
        batch, 0, 2, raw, raw, tokens, positions, torch.tensor([2, 3]), [0, 3, 6]
    )
    trace.file.close()
    rows = [
        json.loads(x)
        for x in next(tmp_path.glob("trace-*.jsonl")).read_text().splitlines()
    ][1:]
    assert len(rows) == 4
    assert rows[0]["ngram3"] == [0, 1, 6]  # slot 2 history positions 8,9 + anchor
    assert rows[1]["ngram3"] == [1, 6, 7]
    assert rows[2]["ngram3"] == [3, 4, 3]  # slot 0, not batch row 1
    assert [r["first_rejection"] for r in rows] == [False, True, False, False]
    for value, old in zip((raw, draft, tokens), before):
        torch.testing.assert_close(value, old, rtol=0, atol=0)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_stale_buffer_is_rejected(tmp_path, monkeypatch):
    trace, batch, _ = setup_trace(tmp_path, monkeypatch)
    trace.stamps[2, 0] = 99
    with pytest.raises(RuntimeError, match="Stale/misaligned"):
        trace.record(
            batch,
            0,
            1,
            torch.zeros(3, 8),
            torch.zeros(3, 8),
            torch.tensor([6, 7, 1]),
            torch.tensor([10, 11, 12]),
            torch.tensor([2]),
            [0, 3],
        )
    trace.file.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_trace_copy_graph_replay_strided_mapping_and_padding():
    from vllm.v1.worker.gpu.spec_decode.ngram_trace_cache import cache_trace_logits

    device = "cuda"
    logits = torch.randn(3, 16, device=device)
    mapping = torch.tensor([[2, 2], [0, 0], [-1, -1]], device=device)[:, 1]
    pos = torch.tensor([[10, 11], [20, 21], [0, 0]], device=device)[:, 1]
    cache = torch.zeros(3, 2, 16, device=device)
    off_logits = logits.neg().clone()
    off_cache = torch.zeros_like(cache)
    stamps = torch.full((3, 2), -1, dtype=torch.int64, device=device)
    off_stamps = stamps.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cache_trace_logits(logits, mapping, pos, cache, stamps, 1)
        cache_trace_logits(off_logits, mapping, pos, off_cache, off_stamps, 1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache_trace_logits(logits, mapping, pos, cache, stamps, 1)
        cache_trace_logits(off_logits, mapping, pos, off_cache, off_stamps, 1)
    logits.add_(1)
    off_logits.sub_(2)
    pos.add_(5)
    graph.replay()
    torch.testing.assert_close(cache[2, 1], logits[0])
    torch.testing.assert_close(cache[0, 1], logits[1])
    assert stamps[2, 1].item() == 16 and stamps[0, 1].item() == 26
    assert cache[1].count_nonzero().item() == 0
    torch.testing.assert_close(off_cache[2, 1], off_logits[0])
    torch.testing.assert_close(off_cache[0, 1], off_logits[1])
    torch.testing.assert_close(off_stamps, stamps)
    assert off_cache[1].count_nonzero().item() == 0


def test_chunk_offset_fixed_residual_and_unreached_suffix(tmp_path, monkeypatch):
    trace, batch, _ = setup_trace(tmp_path, monkeypatch)
    trace.candidates[((4, 3), 0)] = [0, 7]
    raw = torch.zeros(3, 8)
    trace.record(
        batch,
        1,
        2,
        raw,
        raw,
        torch.tensor([3, 4, 5]),
        torch.tensor([5, 6, 7]),
        torch.tensor([1]),
        [0, 3],
    )
    trace.file.close()
    rows = [
        json.loads(x)
        for x in next(tmp_path.glob("trace-*.jsonl")).read_text().splitlines()
    ][1:]
    assert [x["reached"] for x in rows] == [True, False]
    assert [x["first_rejection"] for x in rows] == [True, False]
    expected = (raw[0].softmax(-1) - trace.logits[0, 0].softmax(-1))[[0, 7]]
    torch.testing.assert_close(torch.tensor(rows[0]["fixed"][0]["residual"]), expected)


@pytest.mark.parametrize(
    "temperature,probabilistic,expected",
    [
        (1.0, True, 0.4),
        (0.5, True, 0.04 / 0.38 + 0.2),
        (1.0, False, 0.2),
        (0.0, True, 0.0),
        (0.0, False, 0.0),
    ],
)
def test_overlap_uses_processed_target_and_actual_proposal(
    temperature, probabilistic, expected
):
    # Target was top-k filtered: its third token is impossible. Draft is NOT
    # truncated to the target support, even when its largest logit is token 1.
    target = torch.tensor([0.8, 0.2, 0.0]).log()
    draft = torch.tensor([0.2, 0.5, 0.3]).log()
    assert module.acceptance_overlap(
        target, draft, temperature, probabilistic
    ) == pytest.approx(expected)


def test_paired_record_checks_off_cache_and_preserves_inputs(tmp_path, monkeypatch):
    trace, batch, draft = setup_trace(tmp_path, monkeypatch)
    trace.paired = True
    off = draft.flip(-1).clone()
    off_stamps = trace.stamps.clone()
    trace.set_draft_buffers(
        NS(
            ngram_trace_logits=draft,
            ngram_trace_stamps=trace.stamps,
            ngram_trace_off_logits=off,
            ngram_trace_off_stamps=off_stamps,
        )
    )
    trace.probabilistic = True
    trace.sampler.sampling_states.temperature.np[:] = 0.7
    raw = torch.zeros(3, 8)
    processed = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -float("inf")]]
    ).repeat(3, 1)
    before = off.clone()
    rng = torch.random.get_rng_state()
    trace.record(
        batch,
        1,
        2,
        raw,
        processed,
        torch.tensor([3, 4, 5]),
        torch.tensor([5, 6, 7]),
        torch.tensor([1]),
        [0, 3],
    )
    rows = [
        json.loads(x)
        for x in next(tmp_path.glob("trace-*.jsonl")).read_text().splitlines()
    ][1:]
    contribution = rows[0]["contribution"]
    expected_on = module.acceptance_overlap(processed[0], draft[0, 0], 0.7, True)
    expected_off = module.acceptance_overlap(processed[0], off[0, 0], 0.7, True)
    assert contribution["delta_a"] == pytest.approx(expected_on - expected_off)
    assert contribution["oracle_a"] == max(expected_on, expected_off)
    torch.testing.assert_close(off, before)
    assert torch.equal(rng, torch.random.get_rng_state())
    off_stamps[0, 0] = 999
    with pytest.raises(RuntimeError, match="Stale/misaligned off"):
        trace.record(
            batch,
            1,
            2,
            raw,
            processed,
            torch.tensor([3, 4, 5]),
            torch.tensor([5, 6, 7]),
            torch.tensor([1]),
            [0, 3],
        )
    trace.file.close()


def test_sequential_pair_keeps_on_tokens_and_uses_on_history(monkeypatch):
    """Exercise the actual sequential method on CPU, replacing only GPU copy.

    Loading the method AST avoids importing the CUDA-only runner on CPU hosts.
    The deliberately opposite off proposals must never enter the next step.
    """
    import ast
    import sys
    from types import ModuleType

    path = ROOT / "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DSparkSpeculator"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_sample_sequential"
    )
    namespace = {
        "torch": torch,
        "engram_history": lambda *args: (torch.tensor([0]), torch.tensor([0])),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    copies = []
    copy_module = ModuleType("vllm.v1.worker.gpu.spec_decode.ngram_trace_cache")

    def copy(logits, mapping, positions, cache, stamps, step):
        copies.append(logits.clone())

    monkeypatch.setattr(copy_module, "cache_trace_logits", copy, raising=False)
    monkeypatch.setitem(sys.modules, copy_module.__name__, copy_module)
    predecessors = []

    def embed(tokens):
        predecessors.append(tokens.clone())
        return torch.nn.functional.one_hot(tokens, 3).float()

    model = NS(
        compute_draft_logits=lambda h: torch.zeros(len(h), 3),
        markov_embed=embed,
        markov_bias=lambda z: z,
    )
    memory = NS(
        query=lambda h: h,
        query_norm=lambda h: h,
        modulation=lambda h: None,
        correction=lambda query, older, previous, prev, mod: torch.tensor(
            [[0.0, 0.0, 5.0]]
        ),
    )
    sampled_logits = []

    def sample(logits, *args):
        sampled_logits.append(logits.clone())
        return logits.argmax(-1)

    state = NS(
        _draft_topk=None,
        num_speculative_steps=2,
        sample_indices=torch.arange(2),
        sample_idx_mapping=torch.zeros(2, dtype=torch.long),
        sample_pos=torch.tensor([5, 6]),
        input_buffers=NS(input_ids=torch.tensor([1])),
        _anchor_idx=torch.tensor([0]),
        engram=memory,
        all_token_ids=None,
        model=model,
        use_confidence_head=False,
        _sample_logits=sample,
        draft_tokens=torch.zeros(1, 2, dtype=torch.long),
        ngram_trace_off_logits=None,
        ngram_trace_off_stamps=None,
    )
    run = namespace["_sample_sequential"]
    run(state, 1, torch.ones(2, 3))
    normal = state.draft_tokens.clone()
    normal_logits = [x.clone() for x in sampled_logits]
    predecessors.clear()
    sampled_logits.clear()
    state.ngram_trace_off_logits = torch.empty(1, 2, 3)
    state.ngram_trace_off_stamps = torch.empty(1, 2, dtype=torch.long)
    run(state, 1, torch.ones(2, 3))
    assert state.draft_tokens.tolist() == normal.tolist() == [[2, 2]]
    assert [x.item() for x in predecessors] == [1, 2]
    assert copies[0].argmax(-1).item() == 1
    assert copies[1].argmax(-1).item() == 2
    for a, b in zip(normal_logits, sampled_logits):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
