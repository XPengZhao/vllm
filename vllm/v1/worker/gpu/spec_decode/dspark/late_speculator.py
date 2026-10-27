# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Variable-length 4+1 drafting with bank execution on a separate CUDA stream."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.dspark_late import (
    LateConditioningStages,
    LateSamplingGraph,
    device_ints,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator

logger = init_logger(__name__)


@dataclass
class RequestContext:
    slot: int
    length: int = 0
    bank: torch.Tensor | None = None
    bank_start: int = 0
    ready: torch.cuda.Event | None = None


def validate_late_config(config):
    spec = config.speculative_config
    hf = spec.draft_model_config.hf_config
    late = hf.dflash_config["late_conditioning"]
    if (
        hf.num_hidden_layers != 5
        or hf.block_size != 7
        or (late.get("bank_size", 12), late.get("max_context_lag", 7))
        not in ((12, 7), (13, 8))
    ):
        raise ValueError(
            "4+1 requires five layers, block_size=7, and bank12/lag7 or bank13/lag8; "
            f"got layers={hf.num_hidden_layers}, block_size={hf.block_size}, "
            f"late_conditioning={late}"
        )
    parallel = config.parallel_config
    if (
        any(
            getattr(parallel, name, 1) != 1
            for name in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "decode_context_parallel_size",
                "prefill_context_parallel_size",
            )
        )
        or spec.draft_parallel_config.tensor_parallel_size != 1
    ):
        raise ValueError("4+1 overlap currently requires TP=PP=DP=CP=1")
    if config.scheduler_config.async_scheduling:
        raise ValueError("4+1 requires --no-async-scheduling")
    if getattr(parallel, "enable_dbo", False):
        raise ValueError("4+1 does not support dual batch overlap")
    if config.model_config.hf_config.model_type != "qwen3":
        raise ValueError("4+1 currently supports Qwen3 targets only")
    if config.cache_config.enable_prefix_caching:
        raise ValueError("4+1 requires --no-enable-prefix-caching")
    if (
        not 1 <= spec.num_speculative_tokens <= hf.block_size
        or spec.enable_adaptive_verification
        or spec.num_speculative_tokens_per_batch_size is not None
    ):
        raise ValueError(
            "4+1 requires a fixed proposal length between 1 and block_size (7)"
        )
    if config.model_config.quantization or spec.quantization:
        raise ValueError("4+1 requires unquantized target and draft weights")
    if config.lora_config is not None or config.model_config.supports_multimodal_inputs:
        raise ValueError("4+1 currently supports text-only models without LoRA")
    if getattr(config, "kv_transfer_config", None) is not None:
        raise ValueError("4+1 requires local prefill, without KV transfer")
    if getattr(config, "watermark_config", None) is not None:
        raise ValueError("4+1 does not support watermarking")
    nested = hf.dflash_config
    if (
        nested.get("markov_head_type", "vanilla") != "vanilla"
        or any(
            nested.get(key, {}).get("enabled", False)
            for key in ("moe", "attention_moe")
        )
        or hf.attention_bias
        or hf.hidden_act != "silu"
        or any(t != "full_attention" for t in hf.layer_types)
        or hf.rope_parameters.get("rope_type", "default") != "default"
        or getattr(hf, "is_neox_style", True) is not True
    ):
        raise ValueError("4+1 requires dense vanilla Qwen3 and default RoPE")
    if spec.dspark_draft_topk is not None or getattr(hf, "dspark_draft_topk", None):
        raise ValueError("4+1 uses the full vocabulary, without draft top-k")


class LateDSparkSpeculator(DraftModelSpeculator):
    # Reuse the existing greedy/probabilistic sampler and rejection distribution.
    _sample_logits: Callable[..., torch.Tensor] = DSparkSpeculator._sample_logits

    def __init__(self, config, device):
        validate_late_config(config)
        super().__init__(config, device)
        late = self.draft_model_config.hf_config.dflash_config["late_conditioning"]
        self.overlap = late.get("overlap", True)
        if not isinstance(self.overlap, bool):
            raise ValueError("late_conditioning.overlap must be a boolean")
        self.stream = torch.cuda.Stream(device=device)
        self.contexts: dict[str, RequestContext] = {}
        self._step_cols = torch.arange(
            self.num_speculative_steps, dtype=torch.int32, device=device
        )
        self._d2t_scatter_index = None
        self._draft_scatter_buf = None
        self.num_query_per_req = self.num_speculative_steps
        self.pending_jobs = []
        self.sampling_graphs = {}
        self.graph_enabled = False
        self._fast_decode = False
        self._gpu_lengths_dirty = False
        self.gpu_lengths = torch.zeros(
            self.max_num_reqs, device=device, dtype=torch.long
        )

    def load_draft_model(self, target_model, target_attn_layer_names):
        model = load_dspark_model(target_model, self.vllm_config)
        if model.draft_id_to_target_id is not None:
            raise ValueError("4+1 requires the full target vocabulary")
        self.stages = LateConditioningStages(
            model,
            self.max_num_reqs,
            self.max_model_len,
            self.device,
            self.dtype,
            num_speculative_tokens=self.num_speculative_steps,
        )
        logger.info(
            "DSpark 4+1: bank%d, draft%d, overlap=%s, reserved context K/V %.2f GiB",
            self.stages.bank_size,
            self.num_speculative_steps,
            self.overlap,
            self.stages.kv.numel() * self.stages.kv.element_size() / 2**30,
        )
        # Stages own their context K/V. Do not allocate an unused paged cache
        # for the regular Attention modules whose weights they reuse.
        registry = self.vllm_config.compilation_config.static_forward_context
        for layer in model.model.layers:
            registry.pop(layer.self_attn.attn.layer_name)
        return model

    def set_attn(self, *args, **kwargs):
        pass  # Packed attention does not need paged-attention metadata.

    def init_cudagraph_manager(self, cudagraph_mode):
        self.graph_enabled = (
            cudagraph_mode.name != "NONE"
            and not self.draft_model_config.enforce_eager
            and not self.speculative_config.enforce_eager
        )

    @torch.inference_mode()
    def capture(self):
        if not self.graph_enabled or self.sampling_graphs:
            return
        self.stages.capture_graphs()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        for batch in reversed(self.stages.graph_batches):
            stream.wait_stream(torch.cuda.current_stream(self.device))
            self.sampling_graphs[batch] = LateSamplingGraph(
                self._sample_hidden,
                batch,
                self.hidden_size,
                self.device,
                self.dtype,
                pool,
                stream,
                num_speculative_tokens=self.num_speculative_steps,
            )
        stream.synchronize()
        logger.info(
            "DSpark 4+1 captured %d stage, %d context and %d sampling graphs",
            len(self.stages.graphs),
            len(self.stages.context_graphs),
            len(self.sampling_graphs),
        )

    def clear_cudagraphs(self):
        self.sampling_graphs.clear()
        self.stages.graphs.clear()
        self.stages.context_graphs.clear()
        for name in ("context_weight", "context_norm", "context_eps"):
            if hasattr(self.stages, name):
                delattr(self.stages, name)

    def _sample_hidden(self, hidden, mapping, positions, bonus):
        logits = self.model.compute_draft_logits(hidden.flatten(0, 1)).view(
            hidden.shape[0], hidden.shape[1], -1
        )
        prev, tokens = bonus, []
        for step in range(hidden.shape[1]):
            corrected = logits[:, step] + self.model.markov_bias(
                self.model.markov_embed(prev)
            )
            prev = self._sample_logits(corrected, mapping, positions + step + 1, step)
            tokens.append(prev)
        return torch.stack(tokens, dim=1)

    def remove_request(self, req_id):
        state = self.contexts.pop(req_id, None)
        if state is not None and state.ready is not None:
            assert state.bank is not None
            torch.cuda.current_stream(self.device).wait_event(state.ready)
            state.bank.record_stream(torch.cuda.current_stream(self.device))

    @torch.inference_mode()
    def begin_verification(self, input_batch):
        """Snapshot old-prefix jobs and the event before target forward."""
        main = torch.cuda.current_stream(self.device)
        drafts = input_batch.num_draft_tokens_per_req
        self._fast_decode = bool(
            getattr(self, "sampling_graphs", {})
            and self.stages.context_graphs
            and (self.stages.bank_size == 13 or self.num_speculative_steps <= 6)
            and drafts is not None
            and all(int(n) == self.num_speculative_steps for n in drafts)
            and all(
                int(b - a) == self.num_speculative_steps + 1
                for a, b in zip(
                    input_batch.query_start_loc_np, input_batch.query_start_loc_np[1:]
                )
            )
            and all(req in self.contexts for req in input_batch.req_ids)
        )
        if not self._fast_decode and getattr(self, "_gpu_lengths_dirty", False):
            states = list(self.contexts.values())
            slots = device_ints([s.slot for s in states], self.device)
            for cached_state, length in zip(
                states, self.gpu_lengths[slots].cpu().tolist()
            ):
                cached_state.length = length
            self._gpu_lengths_dirty = False
        if self._fast_decode:
            for i, req in enumerate(input_batch.req_ids):
                if self.contexts[req].slot != int(input_batch.idx_mapping_np[i]):
                    raise ValueError("4+1 request slot changed without reset")
            slots = input_batch.idx_mapping.long()
            old = self.gpu_lengths[slots]
            self._fast_inputs = (slots, old, old + 2)
            # CPU counts are upper bounds; only select capacity with them.
            self._fast_capacity = min(
                self.max_model_len,
                int(max(input_batch.num_computed_tokens_np))
                + self.num_speculative_steps
                + 1,
            )
            torch._assert_async(
                torch.all(self._fast_inputs[1] <= self._fast_capacity),
                "4+1 prefix exceeds graph capacity",
            )
        jobs = []
        for i, req_id in enumerate(input_batch.req_ids):
            state = self.contexts.get(req_id)
            if state is None:
                continue  # Prefill/bootstrap has no prior bank to overlap.
            # The CPU token count is an upper bound. Use our committed prefix
            # here and validate against target GPU positions after verification.
            if (
                input_batch.num_draft_tokens_per_req is None
                or not input_batch.num_draft_tokens_per_req[i]
            ):
                continue  # Chunked prefill must bootstrap from its full prefix.
            jobs.append(state)
        self.pending_jobs = jobs
        self.prefix_ready = torch.cuda.Event()
        self.prefix_ready.record(main)

    @torch.inference_mode()
    def launch_verification_bank(self):
        """Target kernels are enqueued; depend only on the pre-forward event."""
        jobs, self.pending_jobs = self.pending_jobs, []
        if not jobs:
            return
        stream = self.stream if self.overlap else torch.cuda.current_stream(self.device)
        stream.wait_event(self.prefix_ready)
        with (
            torch.cuda.stream(stream),
            torch.autograd.profiler.record_function("dspark4plus1.prepare_bank"),
        ):
            if any(state.bank is not None for state in jobs):
                raise RuntimeError("4+1 bank from the previous round was not consumed")
            if self._fast_decode:
                graph = self.stages.select_graph(len(jobs), self._fast_capacity, False)
                banks = graph.replay_tensors(*self._fast_inputs)
            else:
                for state in jobs:
                    state.bank_start = state.length + 2
                banks = self.stages.prepare_bank_batch(
                    [state.slot for state in jobs],
                    [state.length for state in jobs],
                    [state.bank_start for state in jobs],
                )
            ready = torch.cuda.Event()
            ready.record(stream)
            for state, bank in zip(jobs, banks.unbind(0)):
                state.bank = bank
                state.ready = ready

    def _propose_gpu_decode(
        self,
        input_batch,
        aux,
        num_sampled,
        num_rejected,
        last_sampled,
        temperature,
        seeds,
    ):
        slots, old, bank_starts = self._fast_inputs
        width = self.num_speculative_steps + 1
        starts = input_batch.positions[
            input_batch.query_start_loc[: input_batch.num_reqs].long()
        ]
        counts = width - num_rejected
        lengths = old + counts
        torch._assert_async(
            torch.all(
                (starts == old)
                & (counts >= 1)
                & (counts <= width)
                & (lengths <= self._fast_capacity)
            ),
            "4+1 decode cursor or accepted length is invalid",
        )
        self._copy_request_inputs(
            input_batch.num_reqs, input_batch.idx_mapping, temperature, seeds
        )
        bucket = next(n for n in self.stages.graph_batches if n >= slots.numel())
        features = torch.cat(aux, dim=-1)[: slots.numel() * width]
        # This writes only [old, new). The background bank reads only [0, old).
        self.stages.context_graphs[bucket].replay(features, slots, old, counts)
        main = torch.cuda.current_stream(self.device)
        states = [self.contexts[req] for req in input_batch.req_ids]
        for event in {s.ready for s in states}:
            main.wait_event(event)
        for state in states:
            assert state.bank is not None
            state.bank.record_stream(main)
        banks = torch.stack([s.bank for s in states])
        bonus = last_sampled[slots, 0]
        graph = self.stages.select_graph(slots.numel(), self._fast_capacity, True)
        hidden = graph.replay_tensors(
            slots, lengths, bank_starts, banks, self.model.model.embed_tokens(bonus)
        )
        mapping = torch.where(num_sampled > 0, input_batch.idx_mapping, -1)
        tokens = self.sampling_graphs[bucket].replay(hidden, mapping, lengths, bonus)
        output = self.draft_tokens[: slots.numel()]
        output.copy_(torch.where((num_sampled > 0)[:, None], tokens, -1))
        self.gpu_lengths[slots] = lengths
        self._gpu_lengths_dirty = True
        for state in states:
            state.bank = state.ready = None
        return output

    @torch.inference_mode()
    def propose(
        self,
        input_batch,
        attn_metadata,
        slot_mappings,
        last_hidden_states,
        aux_hidden_states,
        num_sampled,
        num_rejected,
        last_sampled,
        next_prefill_tokens,
        temperature,
        seeds,
        dp_sync=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        mm_inputs=None,
        is_profile=False,
        num_speculative_tokens=None,
    ):
        if dummy_run:
            # Pool memory was reserved at load. Exercise both stages so memory
            # profiling includes temporary attention/bank activations as well.
            feature_width = self.model.model.fc.weight.shape[1]
            batch = max(1, input_batch.num_reqs)
            length = max(1, min(input_batch.num_tokens // batch, self.max_model_len))
            slots, lengths = list(range(batch)), [length] * batch
            features = torch.zeros(
                batch * length, feature_width, device=self.device, dtype=self.dtype
            )
            self.stages.append_context_batch(slots, [0] * batch, features, lengths)
            banks = self.stages.prepare_bank_batch(slots, lengths, [length + 1] * batch)
            hidden = self.stages.finish_batch(
                slots,
                lengths,
                banks,
                [length + 1] * batch,
                features[:batch, : self.hidden_size],
            )
            self.model.compute_draft_logits(hidden.flatten(0, 1))
            return self.draft_tokens[: input_batch.num_reqs].fill_(0)
        if not aux_hidden_states:
            raise ValueError("4+1 requires target auxiliary features")
        if getattr(self, "_fast_decode", False):
            return self._propose_gpu_decode(
                input_batch,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                temperature,
                seeds,
            )
        main = torch.cuda.current_stream(self.device)
        # Resolve scalar lengths once after verification. Bank execution is
        # already in flight and can overlap both target forward and sampling.
        starts = input_batch.positions[
            input_batch.query_start_loc[: input_batch.num_reqs].long()
        ]
        counts = torch.stack((num_sampled, num_rejected, starts)).cpu().tolist()
        features = torch.cat(aux_hidden_states, dim=-1)
        self._copy_request_inputs(
            input_batch.num_reqs, input_batch.idx_mapping, temperature, seeds
        )
        output = self.draft_tokens[: input_batch.num_reqs]
        output.fill_(-1)
        updates, active, rebuild = [], [], []
        pending_reads = []
        for i, req_id in enumerate(input_batch.req_ids):
            slot = int(input_batch.idx_mapping_np[i])
            start = counts[2][i]
            left, right = map(int, input_batch.query_start_loc_np[i : i + 2])
            state = self.contexts.setdefault(req_id, RequestContext(slot))
            if state.slot != slot or state.length != start:
                raise ValueError(
                    "4+1 request slot/cursor changed without reset: "
                    f"request={req_id}, cached_slot={state.slot}, target_slot={slot}, "
                    f"cached_prefix={state.length}, target_position={start}"
                )
            end = right - counts[1][i]
            if not left <= end <= right:
                raise ValueError("invalid 4+1 accepted-context length")
            if state.ready is not None:
                assert state.bank is not None
                pending_reads.append((state.ready, state.bank))
            updates.append((slot, start, left, end))
            state.length += end - left
            if not counts[0][i]:
                state.bank = state.ready = None
                continue  # Non-final chunked prefill.
            if self.stages.bank_size == 12 and state.length - start > 7:
                # Legacy training covers lag 0..7. After full acceptance,
                # rebuild with the committed prefix using its lag-zero path.
                state.bank = state.ready = None
            if state.bank is None:
                state.bank_start = state.length + 1
                rebuild.append(state)
            active.append((i, state))
        if updates:
            if hasattr(self, "gpu_lengths"):
                slots = device_ints([u[0] for u in updates], self.device)
                self.gpu_lengths[slots] = device_ints(
                    [start + end - left for _, start, left, end in updates], self.device
                )
            self.stages.append_context_batch(
                [u[0] for u in updates],
                [u[1] for u in updates],
                torch.cat([features[left:end] for _, _, left, end in updates]),
                [end - left for _, _, left, end in updates],
            )
        for event in {event for event, _ in pending_reads}:
            main.wait_event(event)
        for _, bank in pending_reads:
            bank.record_stream(main)
        if rebuild:
            banks = self.stages.prepare_bank_batch(
                [s.slot for s in rebuild],
                [s.length for s in rebuild],
                [s.bank_start for s in rebuild],
            )
            for state, bank in zip(rebuild, banks.unbind(0)):
                state.bank = bank
        if not active:
            return output
        with torch.autograd.profiler.record_function("dspark4plus1.finish"):
            batch_rows = device_ints([i for i, _ in active], self.device)
            mapping = input_batch.idx_mapping[batch_rows]
            bonus = last_sampled[mapping.long(), 0]
            lengths = [s.length for _, s in active]
            hidden = self.stages.finish_batch(
                [s.slot for _, s in active],
                lengths,
                torch.stack([s.bank for _, s in active]),
                [s.bank_start for _, s in active],
                self.model.model.embed_tokens(bonus),
            )
            positions = device_ints(lengths, self.device)
            graphs = self.sampling_graphs
            if graphs:
                bucket = next(n for n in self.stages.graph_batches if n >= len(active))
                tokens = graphs[bucket].replay(hidden, mapping, positions, bonus)
            else:
                tokens = self._sample_hidden(hidden, mapping, positions, bonus)
            output[batch_rows] = tokens
        for _, state in active:
            state.bank = state.ready = None
        return output
