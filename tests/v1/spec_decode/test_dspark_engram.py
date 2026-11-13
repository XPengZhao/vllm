# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.qwen3_dspark_engram import (
    DSparkEngramMemory,
    engram_history,
)


def test_engram_reordered_history_and_hash_tables():
    config = SimpleNamespace(hidden_size=9, markov_rank=4, rms_norm_eps=1e-6)
    counts = [31, 37, 41, 43]
    module = DSparkEngramMemory(
        config,
        dict(
            hash_version="xor_v1",
            bucket_counts=counts,
            hash_multipliers=[73856093, 19349663, 83492791],
            head_dim=2,
        ),
    )
    tables = [torch.randn(n, 2) for n in counts]
    weights = dict(module.state_dict())
    del weights["table.weight"]
    weights.update({f"tables.{i}.weight": v for i, v in enumerate(tables)})
    module.pack_weights(weights, "")
    module.load_state_dict(weights)
    history = torch.tensor([[37, 80, 90, 22], [97, 60, 10, 42]])
    rows = torch.tensor([1, 0, 1, -1])
    anchors = torch.tensor([3, 1, 0, 0])
    older, previous = engram_history(history, rows, anchors)
    assert older.tolist() == [60, -1, -1, -1]
    assert previous.tolist() == [10, 37, -1, -1]
    current = torch.tensor([42, 80, 97, 0], dtype=torch.int32)
    q = module.query_norm(module.query(torch.randn(4, 9)))
    expected_rows = []
    for a, b, c in zip(older.tolist(), previous.tolist(), current.tolist()):
        bi = (c * 73856093) ^ (b * 19349663)
        tri = bi ^ (a * 83492791)
        expected_rows.append(
            torch.cat(
                [
                    table[h % n] if valid else torch.zeros(2)
                    for table, n, h, valid in zip(
                        tables,
                        counts,
                        [bi, bi, tri, tri],
                        [b >= 0, b >= 0, a >= 0, a >= 0],
                    )
                ]
            )
        )
    mem = torch.stack(expected_rows)
    gate = (q * module.key_norm(module.key(mem))).sum(-1, keepdim=True) / 8**0.5
    torch.testing.assert_close(
        module.correction(q, older, previous, current),
        gate.sigmoid() * module.value(mem),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
def test_engram_graph_reloads_history():
    device = "cuda"
    history = torch.arange(24, device=device).reshape(3, 8)
    rows = torch.tensor([2, 0], device=device)
    anchors = torch.tensor([6, 1], device=device)
    memory = DSparkEngramMemory(
        SimpleNamespace(hidden_size=8, markov_rank=4, rms_norm_eps=1e-6),
        dict(
            hash_version="xor_v1",
            bucket_counts=[31, 37, 41, 43],
            hash_multipliers=[73856093, 19349663, 83492791],
            head_dim=2,
        ),
    ).to(device)
    query = torch.randn(2, 8, device=device)
    current = torch.tensor([5, 7], device=device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        a, b = engram_history(history, rows, anchors)
        captured = memory.correction(query, a, b, current)
    rows.copy_(torch.tensor([1, -1], device=device))
    anchors.copy_(torch.tensor([3, 0], device=device))
    history[1, 2] = 99
    graph.replay()
    assert a.tolist() == [9, -1] and b.tolist() == [99, -1]
    torch.testing.assert_close(captured, memory.correction(query, a, b, current))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.parametrize("topk", [None, 8])
@torch.inference_mode()
def test_seven_step_graph_matches_eager_after_request_reuse(topk):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    # Exercise the real sampling loop and Triton logits-cache scatter. A tiny
    # head replaces the backbone and TP vocabulary projection in this test.
    dev, dtype = "cuda", torch.bfloat16
    b, n, vocab, hidden = 3, 7, 32, 16
    state = DSparkSpeculator.__new__(DSparkSpeculator)
    state.engram = DSparkEngramMemory(
        SimpleNamespace(hidden_size=hidden, markov_rank=8, rms_norm_eps=1e-6),
        dict(
            hash_version="xor_v1",
            bucket_counts=[31, 37, 41, 43],
            hash_multipliers=[73856093, 19349663, 83492791],
            head_dim=4,
        ),
    ).to(device=dev, dtype=dtype)
    lm, w2 = [
        torch.nn.Linear(width, vocab, bias=False, device=dev, dtype=dtype)
        for width in (hidden, 8)
    ]
    w1 = torch.nn.Embedding(vocab, 8, device=dev, dtype=dtype)
    state.model = SimpleNamespace(
        compute_draft_logits=lm,
        markov_embed=w1,
        markov_bias=w2,
        compute_confidence=lambda h, e: e.float().mean(-1).sigmoid(),
        apply_markov_bias_gathered=lambda e, logits, values, ix: logits.scatter_(
            1, ix, values + w2(e).gather(1, ix)
        ),
    )
    state._draft_topk = topk
    state.num_speculative_steps = n
    state.use_confidence_head = True
    state.acceptance_estimator = None
    state._d2t_scatter_index = None
    state.draft_watermarker = None
    state.use_fp64_gumbel = False
    state._step_cols = torch.arange(n, device=dev, dtype=torch.int32)
    state.temperature = torch.ones(4, device=dev)
    state.seeds = torch.arange(4, device=dev)
    state.draft_tokens = torch.zeros(b, n, dtype=torch.int64, device=dev)
    state.draft_logits = torch.zeros(4, n, vocab, dtype=dtype, device=dev)
    state.draft_token_confidence_probs = torch.zeros(b, n, device=dev)
    state.all_token_ids = torch.randint(vocab, (4, 32), device=dev, dtype=torch.int32)
    state.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(b * n, device=dev, dtype=torch.int32)
    )
    state._anchor_idx = torch.arange(b, device=dev) * n
    h = torch.randn(b * n, hidden, device=dev, dtype=dtype)
    # Same initial values as DFlashSpeculator.capture: no live samples.
    state.sample_indices = torch.zeros(b * n, device=dev, dtype=torch.long)
    state.sample_pos = torch.zeros_like(state.sample_indices)
    state.sample_idx_mapping = torch.full((b * n,), -1, device=dev, dtype=torch.int32)
    state._sample_sequential(b, h)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        state._sample_sequential(b, h)
    for rows, anchors in [([2, 0, -1], [3, 1, 0]), ([0, 3, 2], [0, 11, 6])]:
        req = torch.tensor(rows, device=dev)
        pos = torch.tensor(anchors, device=dev)
        live = req >= 0
        state.all_token_ids.random_(0, vocab)
        h.normal_()
        state.input_buffers.input_ids[state._anchor_idx] = state.all_token_ids[
            req.clamp_min(0), pos
        ]
        state.sample_indices.copy_(
            torch.where(
                live[:, None], torch.arange(b * n, device=dev).view(b, n), 0
            ).flatten()
        )
        state.sample_idx_mapping.copy_(req[:, None].expand(-1, n).flatten())
        state.sample_pos.copy_(
            torch.where(live[:, None], pos[:, None] + 1 + state._step_cols, 0).flatten()
        )
        state.draft_logits.fill_(111)
        graph.replay()
        actual = [
            x.clone()
            for x in (
                state.draft_logits,
                state.draft_tokens,
                state.draft_token_confidence_probs,
            )
        ]
        state.draft_logits.fill_(111)
        state._sample_sequential(b, h)
        for captured, eager in zip(
            actual,
            (
                state.draft_logits,
                state.draft_tokens,
                state.draft_token_confidence_probs,
            ),
        ):
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
        assert torch.all((actual[0][req[live]] != 111).any(dim=-1))
        unused = sorted(set(range(4)) - set(rows))
        assert torch.all(actual[0][unused] == 111)
