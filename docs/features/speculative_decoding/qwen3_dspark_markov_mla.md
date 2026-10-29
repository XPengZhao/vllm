# Qwen3-4B DSpark backbone MLA + Markov MLA

This adapter loads the SpecForge `codex/dspark-v1.5-markov-mla` export.
It selects the MLA implementation when `draft_attention_type` is `mla` and
uses an MLA head only when `dflash_config.markov_head_type` is `mla`.
Backbone-only MLA exports with `markov_head_type: vanilla` use the original
`W1(previous token) -> W2` head, including older offline SpecForge exports.
Backbone and Markov projection dimensions are read from the exported config. Missing trainable
weights are errors; the frozen embedding and LM head may be shared with Qwen3.

## Execution

The backbone keeps its decoder residuals, RMSNorms and FFNs. The Markov head is
`W1(previous token) -> MLA -> W2`, without a residual, FFN or output norm.
Its KV latent RMSNorm has a loaded weight. Confidence uses the original W1
embedding together with the backbone output.

Each attention module independently projects the normalized, fused target
features. Expanded K/V tensors use vLLM's paged caches; compressed latent
caching is not implemented. With 32 heads, width 128, BF16, and six attention
modules, draft KV uses 98,304 bytes per cached token before allocator padding
and overhead (about 384 MiB for 4,096 tokens in one request).

For backbone-only MLA, only the five backbone caches are allocated and
updated: 81,920 bytes per cached token (320 MiB for 4,096 tokens). The vanilla
head has no KV cache or attention metadata; sequential sampling and CUDA Graph
capture use the original Markov path. The same serving command applies with
the backbone-only checkpoint path, without changing its exported head type.

The backbone inserts its query block into its cache. The head only reads its
own target-feature cache, with visible length equal to the anchor position.
Every head step has one query per request at `anchor + step`; the sampled token
is at `anchor + step + 1`. Future, rejected, and padded positions are excluded.
Accepted target features are projected once per update and reused at every
head step. Block allocation and request lifecycle stay with vLLM.

## Initial supported configuration

Use unquantized BF16, full vocabulary, TP/PP/CP 1, non-causal full attention,
NeoX RoPE, `cache_dtype=auto`, and FlashAttention. Draft top-k is unsupported.
Start in eager mode; validate CUDA Graph replay on the deployment GPU before
removing `--enforce-eager`. The graph path uses fixed query-offset buffers,
one-query head metadata, no AOT schedule, and one FlashAttention split.

Replace the checkpoint path with the trained export directory containing
`config.json` and weights:

```bash
vllm serve /data1/models/Qwen3-4B \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --enforce-eager \
  --speculative-config '{
    "method": "dspark",
    "model": "/path/to/exported-markov-mla-checkpoint",
    "num_speculative_tokens": 7,
    "draft_sample_method": "greedy",
    "attention_backend": "FLASH_ATTN",
    "enable_adaptive_verification": false
  }'
```

For probabilistic draft sampling use `"draft_sample_method": "probabilistic"`.
This additionally stores full-vocabulary draft probabilities. Start with
adaptive verification disabled to measure the head without threshold effects.

## Validation

In a configured vLLM GPU development environment:

```bash
.venv/bin/python -m pytest tests/models/test_dspark_mla.py -k qwen -q
.venv/bin/python -m pytest tests/v1/worker/test_dspark_utils.py -q
.venv/bin/python -m pytest tests/v1/spec_decode/test_dspark_topk.py -q
```

The projection tests cover both query widths, shared rotary keys, trainable
latent normalization, incremental context projection, no head cache writes,
no head residual, and prefix metadata isolation. They do not substitute for
GPU cache or end-to-end verification.

Before reporting inference readiness, compare exported SpecForge/vLLM hidden
states and logits on identical inputs, then check eager versus graph outputs
across varying batch sizes, chunked prefills, rejected proposals, and request
reuse. Run the existing DSpark GSM8K evaluation with the trained checkpoint;
record accuracy, acceptance length/rate, throughput and peak memory against
`dspark-v1.4` using the same prompts, sampling and hardware. There is no measured
acceptance or performance result for this adapter yet.
