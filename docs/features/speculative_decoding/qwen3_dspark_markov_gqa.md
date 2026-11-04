# Qwen3-4B DSpark GQA backbone + Markov MHA

Loads SpecForge exports from `codex/dspark-v1.5-markov-gqa`. The branch and
`markov_head_type` retain the GQA name; the current training configuration uses
8 query heads, 8 KV heads and head dimension 128 (MHA).

The checkpoint must contain:

```json
{
  "dflash_config": {
    "markov_head_type": "gqa",
    "markov_rank": 256,
    "markov_gqa": {
      "num_attention_heads": 8,
      "num_key_value_heads": 8,
      "head_dim": 128
    }
  }
}
```

The five-layer GQA backbone is unchanged. The head computes
`W1(previous token) -> Q/K RMSNorm + RoPE + attention -> O -> W2`,
without a residual, FFN or output norm. K/V come from the same normalized
projected target features as the backbone, through independent weights.
Confidence continues to use the raw W1 embedding and backbone output.
Missing head weights are rejected instead of leaving random parameters.

The head owns a sixth paged KV cache, costing 4 KiB per context token in BF16
(16 MiB at 4096 tokens per request, before allocator overhead). It never writes
query tokens into this cache. Every sequential step reads only positions before
the anchor, with query position `anchor + step`. Existing vLLM request/block
management owns allocation, rejection handling and reuse.

## Serve

The initial implementation supports unquantized BF16, TP/PP/CP=1, full vocabulary,
non-causal full attention, NeoX RoPE and `cache_dtype=auto`. Use FlashAttention.
Draft candidate truncation (`dspark_draft_topk`) is unsupported; ordinary sampling
parameters such as top-k and top-p remain applicable.

No additional Markov-specific serving argument is needed. Select the exported
checkpoint; its config selects the head automatically:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /data1/models/Qwen3-4B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --speculative-config '{
    "method": "dspark",
    "model": "/path/to/exported-markov-mha-checkpoint",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "attention_backend": "FLASH_ATTN",
    "enable_adaptive_verification": false
  }'
```

This command leaves CUDA Graph enabled. The sequential head uses the existing
Markov MLA graph path: fixed query-offset buffers, one query per request,
pre-anchor sequence lengths, disabled cascade/AOT schedule, and one attention
split. Baseline and existing MLA checkpoints still select their original paths.

## Validation

In a configured vLLM environment:

```bash
.venv/bin/python -m pytest tests/models/test_dspark_mla.py -k qwen -q
.venv/bin/python -m pytest tests/v1/worker/test_dspark_utils.py -q
.venv/bin/python -m pytest tests/v1/spec_decode/test_dspark_topk.py -q
```

The tests cover the training attention formula for MHA and GQA, incremental
projection, separate cache writes, read-only head attention, projection weight
mapping, missing head weights, prefix metadata and sequential token positions.
Local validation used isolated CPU execution because the development machine
lacks a complete vLLM installation and CUDA. It does not validate GPU kernels,
CUDA Graph replay or end-to-end checkpoint loading.

Before drawing acceptance or performance conclusions, compare the exported
checkpoint against training logits, then compare eager and graph runs with
varying batches, chunked prefills, rejected suffixes and reused request slots.
