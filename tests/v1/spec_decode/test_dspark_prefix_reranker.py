# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for prefix attention and the block-local greedy draft walk.

Unlike the gathered-Markov top-k tests, these test a learned causal prefix and
candidate scoring after the dense Markov correction. No model/GPU is required.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.models.dspark_prefix_reranker import DSparkPrefixReranker


def _reranker(top_k=3):
    torch.manual_seed(17)
    model = DSparkPrefixReranker(6, 4, 4, width=8, num_heads=2, top_k=top_k)
    with torch.no_grad():
        model.residual_out.weight.normal_()
    return model.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cached_prefix_matches_parallel_causal_encoding(dtype):
    model = _reranker().to(dtype=dtype)
    embeddings = torch.randn(2, 4, 4, dtype=dtype)
    parallel, _ = model.encode(embeddings)
    cache = None
    outputs = []
    for i in range(4):
        encoded, cache = model.encode(embeddings[:, i : i + 1], cache)
        outputs.append(encoded)
    tolerance = 0.02 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(
        torch.cat(outputs, dim=1), parallel, atol=tolerance, rtol=tolerance
    )
    assert cache is not None
    assert cache[0].shape == (2, 2, 4, 4)


def test_prefix_does_not_read_future_tokens():
    model = _reranker()
    embeddings = torch.randn(2, 4, 4)
    original, _ = model.encode(embeddings)
    embeddings[:, 2:] += 10
    changed, _ = model.encode(embeddings)
    torch.testing.assert_close(original[:, :2], changed[:, :2])


@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_zero_residual_preserves_greedy_choice_including_ties(top_k):
    model = _reranker(top_k)
    with torch.no_grad():
        model.residual_out.weight.zero_()
    logits = torch.tensor([[1.0] * 7, [0.0, 1.0, 3.0, 2.0, 3.0, 0.0, 0.0]])
    scores, ids = model.score(
        logits, torch.randn(2, 6), torch.randn(2, 8), torch.randn(7, 4)
    )
    selected = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
    torch.testing.assert_close(selected, logits.argmax(-1))
    torch.testing.assert_close(scores, logits.gather(-1, ids))


def test_scores_use_output_token_features_and_fp32_residual_addition():
    model = _reranker().to(dtype=torch.bfloat16)
    logits = torch.randn(2, 7, dtype=torch.bfloat16)
    hidden = torch.randn(2, 6, dtype=torch.bfloat16)
    prefix = torch.randn(2, 8, dtype=torch.bfloat16)
    w2 = torch.randn(7, 4, dtype=torch.bfloat16)
    scores, ids = model.score(logits, hidden, prefix, w2)
    query = model.residual_out(
        model.query(torch.cat([model.hidden_proj(hidden), prefix], -1))
    )
    delta = (query.unsqueeze(-2) * model.candidate_proj(w2[ids])).sum(-1)
    delta = delta / math.sqrt(model.width)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores, logits.gather(-1, ids).float() + delta.float())


def test_greedy_walk_matches_full_prefix_recompute_and_resets_between_rounds():
    model = _reranker()
    w1, w2 = torch.randn(7, 4), torch.randn(7, 4)
    base, hidden = torch.randn(2, 4, 7), torch.randn(2, 4, 6)
    anchors = torch.tensor([2, 5])

    def sample(anchor, base_logits=base, h=hidden):
        return model.sample(
            base_logits,
            h,
            anchor,
            lambda t: F.embedding(t, w1),
            lambda e: F.linear(e, w2),
            w2,
        )

    previous = anchors
    prefix_ids, expected = [], []
    for i in range(4):
        prefix_ids.append(previous)
        prefix, _ = model.encode(F.embedding(torch.stack(prefix_ids, -1), w1))
        logits = base[:, i] + F.linear(F.embedding(previous, w1), w2)
        scores, ids = model.score(logits, hidden[:, i], prefix[:, -1], w2)
        previous = ids.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        expected.append(previous)
    result = sample(anchors)
    torch.testing.assert_close(result, torch.stack(expected, -1))
    sample(torch.tensor([6, 0]))
    torch.testing.assert_close(sample(anchors), result)
    torch.testing.assert_close(
        sample(anchors.flip(0), base.flip(0), hidden.flip(0)), result.flip(0)
    )


def test_markov_correction_precedes_candidate_selection():
    model = _reranker(top_k=1)
    base = torch.tensor([[[10.0, 0.0, 1.0]]])
    w1 = torch.ones(3, 4)
    w2 = torch.tensor([[0.0] * 4, [5.0] * 4, [0.0] * 4])
    result = model.sample(
        base,
        torch.zeros(1, 1, 6),
        torch.tensor([0]),
        lambda t: F.embedding(t, w1),
        lambda e: F.linear(e, w2),
        w2,
    )
    assert result.item() == 1  # Not in the top-1 base-logit candidate set.


def test_cache_rejects_cross_round_overflow_and_multi_token_append():
    model = _reranker()
    _, full_cache = model.encode(torch.randn(1, 4, 4))
    with pytest.raises(ValueError, match="reset cache"):
        model.encode(torch.randn(1, 1, 4), full_cache)
    _, cache = model.encode(torch.randn(1, 1, 4))
    with pytest.raises(ValueError, match="exactly one"):
        model.encode(torch.randn(1, 2, 4), cache)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_graph_replay_uses_new_anchors_and_resets_local_kv():
    model = _reranker().cuda().to(dtype=torch.bfloat16)
    w1, w2 = [torch.randn(7, 4, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    base = torch.randn(2, 4, 7, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(2, 4, 6, device="cuda", dtype=torch.bfloat16)
    anchors = torch.tensor([2, 5], device="cuda")

    def sample():
        return model.sample(
            base,
            hidden,
            anchors,
            lambda t: F.embedding(t, w1),
            lambda e: F.linear(e, w2),
            w2,
        )

    with torch.inference_mode():
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sample()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = sample()
        for new_ids in ([1, 6], [5, 2], [2, 5]):
            anchors.copy_(torch.tensor(new_ids, device="cuda"))
            expected = sample()
            graph.replay()
            torch.testing.assert_close(captured, expected)
