# Qwen3.8-Flash-Next DSpark baseline

This adapter runs a SpecForge Qwen3.8-Flash-Next baseline export with the existing
Qwen3 DSpark backbone and vanilla Markov head. It does not include the prefix
reranker. vLLM uses the internal `Qwen4Exp` architecture for this target.

## Hidden features

The target exposes the auxiliary layers requested by the draft checkpoint.
Each auxiliary feature materializes the pending HC combine and averages the
HC streams in FP32, then casts back to the model dtype. Reading an auxiliary
feature leaves the target's pending combine unchanged.

The target's sampled hidden state still comes from its final HC mixer. Its
pre-mixer multi-stream buffer is used only by native MTP, not by DSpark.
Markov and confidence settings in `dflash_config` take precedence over the
corresponding top-level settings, matching the draft backbone's convention.

## Start the baseline service

Use the baseline export, with `Qwen3DSparkModel` as its architecture and both
`markov_head.markov_w1.weight` and `markov_head.markov_w2.weight` in its weights.
The paths below match the Qwen3.8 baseline experiment.

```bash
export CUDA_VISIBLE_DEVICES=1,3,6,7
export CUDA_HOME=/usr/local/cuda-13.0

vllm serve /public/llm_models/Qwen/Qwen3.8-Flash-Next \
  --served-model-name qwen38-flash-next \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --enforce-eager \
  --max-model-len 8192 \
  --block-size 192 \
  --generation-config vllm \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --speculative-config '{
    "method": "dspark",
    "model": "/public/workspace/dspark/exports/qwen38-baseline-lr3e4-step7812",
    "num_speculative_tokens": 7,
    "draft_sample_method": "greedy",
    "enable_adaptive_verification": false
  }' \
  --host 0.0.0.0 \
  --port 8000
```

For a probabilistic baseline draft, change `draft_sample_method` to
`"probabilistic"`. This caches full draft logits for rejection sampling and
requires additional GPU memory. Target sampling parameters are set in each
request. The greedy command above matches the recorded baseline experiment.

## Validation

Run the relevant regression suites in a configured vLLM environment:

```bash
.venv/bin/python -m pytest \
  tests/models/qwen4_exp/test_config.py \
  tests/v1/worker/test_gpu_model_runner_v2.py \
  tests/v1/spec_decode/test_dspark_topk.py -q
```

On the GPU server, first confirm the service loads the baseline checkpoint and
returns a complete response. Then evaluate the baseline with the same GSM8K
prompt, Target sampling parameters, concurrency, and output limit as the
previous run. Record acceptance counters as a before/after difference.
Compare greedy speculative output with a Target-only service on identical
raw-token prefixes, including blocks after an earlier rejection, to check the
continued decoding path. Validate probabilistic drafting separately.

The development Mac has no CUDA. Fourteen isolated CPU regression cases passed
using real FP32/BF16 tensors with vLLM runtime imports stubbed. Full pytest
collection is blocked by missing Python dependencies (`regex` was the first
failure). These checks do not establish GPU model parity or online performance.
