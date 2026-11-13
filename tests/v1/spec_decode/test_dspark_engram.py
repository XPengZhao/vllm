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
