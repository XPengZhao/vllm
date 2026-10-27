# Qwen3-4B DrawSpec: configurable proposals with concurrent bank execution

This branch supports dense 4+1 checkpoints trained with either SpecForge's
v3 `4+1-bank13-draft7-uniform-lag0to8-v3` semantics or the earlier v2
`4+1-shared-mask-uniform-lag0to7-v2` semantics. The proposal length is selected at startup by `num_speculative_tokens`
(1 through 7). Ordinary DSpark continues to use the existing speculator.

Legacy exports containing only `"late_conditioning": {"enabled": true}` use
a twelve-state bank and lag 0..7. They can be served directly without retraining
or changing their config. Explicit `bank_size: 12, max_context_lag: 7` is also
supported. Keep the original bank width: its bidirectional attention affects
every selected state.

## Training and export of new thirteen-state models

Use SpecForge branch `codex/dspark-v1.5-4plus1-overlap` with
`model.draft_model_config=configs/qwen3-4b-dspark-4plus1.json`. It trains a
thirteen-state bank, uniformly sampled lag 0..8, and seven next-token labels.
Starting from the old 12-state run requires a **weights-only warm start** in a
new run: `model.draft_checkpoint_path=/path/to/old-step`. Keep
`training.resume_from` unset for that transition. Optimizer, LR schedule and
data position start anew; a v3 checkpoint subsequently supports exact resume.

Export a newly trained v3 checkpoint (existing legacy exports need no re-export):

```bash
cd "$HOME/SpecForge"
source .venv/bin/activate
specforge export --to hf \
  --checkpoint "$HOME/outputs/qwen3-4b/NEW_RUN/NEW_RUN-step2616" \
  --draft-config configs/qwen3-4b-dspark-4plus1.json \
  --embedding-source /data1/models/Qwen3-4B \
  --output-dir /data1/models/dspark-4plus1-bank13-step2616
```

## Serve

Use the vLLM branch with the same name and its usual installed environment:

```bash
cd "$HOME/vllm"
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=1 vllm serve /data1/models/Qwen3-4B \
  --served-model-name qwen3-4b \
  --port 11443 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dspark","model":"/data1/models/dspark-4plus1-bank13-step2616","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

Supported scope: one GPU, unquantized text Qwen3 target/draft, vanilla Markov
head, a fixed proposal length of 1 through 7 tokens. Greedy and probabilistic drafting use the
existing DSpark sampler and standard vLLM rejection sampling. Adaptive
verification, distributed parallelism, LoRA, watermarking and KV transfer are
rejected. Requests can batch, finish, be preempted and reuse slots. Chunked
prefill is supported; its final chunk bootstraps from the complete prefix.

For an existing legacy export, change only the draft `model` path above, for
example to `$HOME/models/qwen3-4b/dspark-4plus1-20261022-175740-step2616`.
Set `num_speculative_tokens: 6` to use `anchor + 5 states` in the final layer
and propose six tokens. The 12-state bank is unchanged, and full acceptance no
longer requires a synchronous bank rebuild. Sampling buffers and CUDA graphs
use the configured proposal length. No checkpoint edits or retraining are
required. The startup log reports both bank width and proposal length.

Using fewer than seven positions changes final-layer bidirectional attention;
it does not compute seven outputs and truncate them. The original seven-token
behavior is retained with `num_speculative_tokens: 7`.

The draft batches projections and FFNs across requests and uses variable-length
FlashAttention. With CUDA graphs enabled, startup captures the four-layer bank,
the final layer, and the vocabulary head plus the configured number of Markov sampling steps.
Request counts use power-of-two buckets; context capacities use 128, 512, 2048,
and so on, capped at `--max-model-len`. The packed buffers have bucket capacity,
but a bounded Triton packing kernel copies only actual context and self K/V.
Unused capacity is neither gathered nor written, and attention reads only the
actual per-request lengths. Padding cannot enter
another request's attention or sampling cache.

Bank, context-update, final-layer, and sampling graphs have separate memory pools. Bank replay
stays on the background stream, and graph outputs are copied before reuse,
including legacy full-acceptance rebuilds. There is no runtime graph capture.
For uniform decode batches, the 13-state model and legacy models with N<=6
keep committed lengths and acceptance counts on the GPU. A separate context
update graph projects all five layers' K/V in one GEMM and stores only accepted
positions. It writes the new suffix while the background bank reads the old
prefix; the main stream waits for the bank only before consuming its output.
Graph capacity is selected from CPU upper bounds, with cursor checks on the GPU.

Mixed prefill/decode, irregular query lengths, eager mode and legacy N=7 use the
checked fallback. Returning to that path refreshes CPU cursor state once.
Request bookkeeping remains on the CPU; the uniform decode path does not copy
acceptance counts to the CPU. Context attention still packs valid K/V each layer.

No new launch flag is required. Keep target CUDA graphs enabled and omit
`--enforce-eager`. To compare against eager drafting while retaining target
graphs, set `"enforce_eager": true` in `--speculative-config`. The startup log
reports the captured stage and sampling graph counts. Profiling captures are
released before the real startup capture.

The private context K/V pool is reserved at model load; it costs about 1.25 GiB
at 8 requests and 8192 tokens for Qwen3-4B BF16. Graph pools require additional
memory and are included in CUDA graph memory profiling. Set `--max-num-seqs`
and `--max-model-len` explicitly: large defaults reserve excessive memory.
Temporary bank K/V never enter the persistent context cache.

## Dependency and position contract

At the start of a verification anchored at t, the committed context covers
positions <t. The runner records a prefix-ready CUDA event **before** enqueuing
target forward. Once target kernels are enqueued, it launches the four-layer
bank on a separate stream waiting only on that earlier event. This is not a
`wait_stream(main)` after target forward, which would serialize the two.

The bank contains twelve or thirteen pure MASK slots starting at t+2 and
never reads the new bonus or current-block target features. For proposal length
N and r=0..N accepted proposals, the next anchor is b=t+r+1. The selected N-1
states are bank[r:r+N-1], at b+1..b+N-1. The final stage computes
[adapted bonus, N-1 states] and predicts N tokens at b+1..b+N. First-request
prefill uses bank start b+1 with the full prompt prefix.

At N<=6, both bank versions cover every acceptance boundary without rebuilding.
At N=7, a thirteen-state bank also covers every boundary. A legacy twelve-state
bank must rebuild after r=7: the runner waits for its outstanding read, commits
the eight verified context positions, and rebuilds at b+1 using the complete
accepted prefix (the trained lag-zero bootstrap path). Other rounds reuse the
background bank. Serial mode uses the same boundary policy.

Target and draft stages have distinct context K/V and use no shared vLLM
attention/workspace buffers. Request removal waits on outstanding bank reads
before its cache slot can be reused. Fallback batches resolve lengths on the
CPU; the uniform decode path keeps those lengths on the GPU.

## Validate on the server

With both checkouts available and SpecForge on PYTHONPATH:

```bash
cd "$HOME/vllm"
PYTHONPATH="$HOME/SpecForge:$HOME/vllm" .venv/bin/python -m pytest -v \
  tests/models/test_dspark_late.py \
  tests/models/test_dspark_late_runtime.py \
  tests/models/test_dspark_late_e2e.py
```

Stage tests compare actual serving operations against SpecForge training for
bootstrap and every r=0..7 boundary in FP32/BF16. Runtime tests cover reordered
batches, mixed r=0/r=7, chunked prefill and slot reuse. CUDA tests check stage
parity, measured GPU overlap with a synthetic verifier interval, and a real
tiny Qwen3 vLLM engine against target-only greedy outputs, including stochastic
sampling smoke. The synthetic overlap test is not a throughput result for the
real target; run GSM8K on the real exported checkpoint as well.

For a serial scheduling control that preserves the same old-prefix banks, set
`dflash_config.late_conditioning.overlap` to `false` in the **exported draft's
config.json**, then restart the server. Keep every other config field unchanged:

```json
"overlap": false
```

Add this field inside the existing `late_conditioning` object, preserving its
bank/lag fields (or their absence in legacy exports). Restore `overlap` to
`true` (or remove that field) to enable concurrency again.
The serving CLI's global `--hf-overrides` does not override this draft config.

For a GPU timeline, add a torch profiler configuration, call `/start_profile`,
send several warm requests, then call `/stop_profile`:

```bash
--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/user/logs/4plus1-traces","torch_profiler_with_stack":false}'
```

Inspect `dspark4plus1.prepare_bank` and `dspark4plus1.finish` in the worker
trace. Confirm bank kernels run on a different CUDA stream and overlap target
verification kernels; CPU scope overlap alone is insufficient. Compare serial
and concurrent runs under identical prompts, sampling, cache and concurrency
settings. This checkout was developed without a CUDA device: real GPU serving,
overlap duration and end-to-end speedup require the server validation above.
