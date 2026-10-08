# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_dspark
from vllm.model_executor.models.dspark_ngram_markov import DSparkNgramMarkovAdapter
from vllm.v1.worker.gpu.spec_decode.dspark.ngram import (
    DSparkNgramLookup,
    gather_ngram_context,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ngram_adapter_matches_training_formula(dtype):
    torch.manual_seed(7)
    adapter = DSparkNgramMarkovAdapter(6, 4, 1e-6).to(dtype)
    with torch.no_grad():
        adapter.ngram_value_proj.weight.normal_()
    unigram = torch.randn(2, 3, 4, dtype=dtype)
    hidden = torch.randn(2, 3, 6, dtype=dtype)
    raw = torch.randn(2, 3, 6, dtype=dtype)
    normalized = adapter.ngram_norm(raw)
    query = adapter.ngram_query_norm(
        adapter.ngram_query_proj(adapter.ngram_hidden_norm(hidden))
    )
    key = adapter.ngram_key_norm(adapter.ngram_key_proj(normalized))
    score = (query.float() * key.float()).sum(-1, keepdim=True) / math.sqrt(4)
    gate = torch.sigmoid(
        torch.sign(score) * torch.sqrt(score.abs().clamp_min(1e-6))
    ).to(dtype)
    expected = unigram + gate * adapter.ngram_value_proj(normalized)
    torch.testing.assert_close(adapter(unigram, hidden, raw), expected, rtol=0, atol=0)


def test_zero_ngram_value_preserves_vanilla_markov():
    adapter = DSparkNgramMarkovAdapter(6, 4, 1e-6)
    unigram = torch.randn(2, 4)
    torch.testing.assert_close(
        adapter(unigram, torch.randn(2, 6), torch.randn(2, 6)), unigram
    )


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "markov_w1",
        "markov_w2",
        "ngram_query_proj",
        "ngram_key_proj",
        "ngram_value_proj",
    ],
)
def test_exported_ngram_weights_load_or_fail_closed(monkeypatch, missing):
    head = nn.Module()
    head.markov_w1 = nn.Embedding(11, 4)
    head.markov_w2 = nn.Linear(4, 11, bias=False)
    head.ngram_adapter = DSparkNgramMarkovAdapter(6, 4, 1e-6)
    weights = {
        "markov_head." + key.replace("ngram_adapter.", ""): torch.randn_like(value)
        for key, value in head.state_dict().items()
    }
    receiver = SimpleNamespace(
        model=SimpleNamespace(
            markov_head=head,
            ngram_markov_enabled=True,
            prefix_reranker=None,
            confidence_head=None,
            _build_fused_kv_buffers=Mock(),
        ),
        config=SimpleNamespace(vocab_size=11, draft_vocab_size=11),
        target_vocab_size=11,
    )

    class Loader:
        def __init__(self, model):
            assert model is receiver

        def load_weights(self, source, mapper):
            with torch.no_grad():
                for name, value in mapper.apply(source):
                    head.state_dict()[name.removeprefix("model.markov_head.")].copy_(
                        value
                    )

    monkeypatch.setattr(qwen3_dspark, "AutoWeightsLoader", Loader)
    monkeypatch.setattr(qwen3_dspark, "process_eagle_weight", lambda *args: None)
    if missing is not None:
        del weights[f"markov_head.{missing}.weight"]
        with pytest.raises(ValueError, match=f"missing weights:.*{missing}"):
            qwen3_dspark.Qwen3DSparkForCausalLM.load_weights(receiver, weights.items())
        receiver.model._build_fused_kv_buffers.assert_not_called()
    else:
        qwen3_dspark.Qwen3DSparkForCausalLM.load_weights(receiver, weights.items())
        for key, value in head.state_dict().items():
            exported = weights["markov_head." + key.replace("ngram_adapter.", "")]
            torch.testing.assert_close(value, exported, rtol=0, atol=0)
        receiver.model._build_fused_kv_buffers.assert_called_once()


def test_context_reads_before_anchor_and_follows_request_reordering():
    # Suffixes are stale rejected drafts and must not enter the next window.
    ids = torch.tensor([[1, 2, 3, 80, 81], [4, 5, 6, 90, 91]])
    contexts = gather_ngram_context(
        ids, torch.tensor([1, 0, -1]), torch.tensor([3, 1, 4]), 3, 99
    )
    torch.testing.assert_close(
        contexts, torch.tensor([[4, 5, 6], [99, 99, 1], [99, 99, 99]])
    )
    ids[:, 3:] = -123
    torch.testing.assert_close(
        gather_ngram_context(ids, torch.tensor([1, 0]), torch.tensor([3, 1]), 3, 99),
        contexts[:2],
    )


@pytest.mark.parametrize("pinned", [False, True])
def test_lookup_reuses_table_and_does_not_touch_target_prefetch(pinned):
    class Table(nn.Module):
        supports_prefetch = pinned

        def __init__(self):
            super().__init__()
            self.weight = torch.arange(30).view(10, 3).float()
            self.start_prefetch = Mock(side_effect=AssertionError("target buffer"))

        def forward(self, ids):
            assert not pinned
            return self.weight[ids]

        def sync_lookup(self, ids):
            assert pinned
            return self.weight[ids]

        def dequantize(self, values, dtype):
            return (values * 0.5).to(dtype)

    table = Table()
    compute = Mock(return_value=torch.tensor([[1, 2], [3, 4]]))
    lookup = SimpleNamespace(
        embedding_dim=6,
        ngram_size=3,
        eos_token_id=9,
        compute_ngram_ids=compute,
        ngram_embedding=table,
    )
    target = SimpleNamespace(
        model=SimpleNamespace(
            layers=[None, SimpleNamespace(ple=SimpleNamespace(ple_embedding=lookup))]
        )
    )
    adapter = DSparkNgramLookup(target, 6)
    previous, context = torch.tensor([5, 7]), torch.tensor([[1, 2], [3, 4]])
    result = adapter(previous, context, torch.bfloat16)
    torch.testing.assert_close(
        result,
        (table.weight[torch.tensor([[1, 2], [3, 4]])].flatten(-2) * 0.5).bfloat16(),
    )
    args = compute.call_args.args
    torch.testing.assert_close(args[0], previous)
    torch.testing.assert_close(args[1], torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(args[2], context)
    table.start_prefetch.assert_not_called()
