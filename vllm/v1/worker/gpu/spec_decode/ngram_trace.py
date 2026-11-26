# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in eager diagnostics. Never called within a CUDA graph capture."""

import json
import os
from pathlib import Path

import torch


def acceptance_overlap(processed_target, draft_logits, temperature, probabilistic):
    """Expected ordinary rejection acceptance at one fixed history.

    Target logits already include all sampling processors. DSpark proposals
    apply temperature only, or are delta distributions for greedy drafting.
    """
    if temperature == 0:
        return float(processed_target.argmax().item() == draft_logits.argmax().item())
    p = processed_target.float().softmax(-1)
    if not probabilistic:
        return float(p[draft_logits.argmax()].item())
    q = (draft_logits.float() / temperature).softmax(-1)
    return float(torch.minimum(p, q).sum().item())


class NgramTrace:
    def __init__(self, sampler, spec_config):
        if (
            spec_config.method != "dspark"
            or spec_config.enable_adaptive_verification
            or spec_config.rejection_sample_method in ("synthetic", "block")
        ):
            raise ValueError(
                "Ngram trace requires DSpark with ordinary fixed-length verification"
            )
        self.paired = os.environ.get("VLLM_NGRAM_TRACE_PAIRED", "0") == "1"
        self.sampler = sampler
        self.probabilistic = spec_config.draft_sample_method == "probabilistic"
        self.prefix = os.environ.get("VLLM_NGRAM_TRACE_PREFIX", "ngram-")
        if not self.prefix:
            raise ValueError("Ngram trace request prefix must be nonempty")
        self.k = int(os.environ.get("VLLM_NGRAM_TRACE_TOPK", "32"))
        if not 1 <= self.k <= 256:
            raise ValueError("Ngram trace topk must be in [1, 256]")
        self.limit = int(os.environ.get("VLLM_NGRAM_TRACE_MAX_EVENTS", "2000000"))
        if self.limit <= 0:
            raise ValueError("Ngram trace event limit must be positive")
        self.count = 0
        self.candidates = {}
        source = os.environ.get("VLLM_NGRAM_TRACE_CANDIDATES")
        if source:
            for line in Path(source).read_text().splitlines():
                row = json.loads(line)
                self.candidates[(tuple(row["ngram_token_ids"]), row["position"])] = row[
                    "token_ids"
                ]
        directory = Path(os.environ["VLLM_NGRAM_TRACE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        self.file = (directory / f"trace-{os.getpid()}.jsonl").open("x", buffering=1)
        self.file.write(
            json.dumps(
                dict(
                    type="metadata",
                    version=2,
                    paired=self.paired,
                    comparison="same on-trajectory history; remove Engram increment",
                    overlap="processed target vs temperature-only draft proposal",
                    prefix=self.prefix,
                    probabilistic=self.probabilistic,
                    topk=self.k,
                    candidates=source,
                    max_events=self.limit,
                )
            )
            + "\n"
        )
        self.logits = self.stamps = None
        self.off_logits = self.off_stamps = None

    def set_draft_buffers(self, speculator):
        self.logits = getattr(speculator, "ngram_trace_logits", None)
        self.stamps = getattr(speculator, "ngram_trace_stamps", None)
        self.off_logits = getattr(speculator, "ngram_trace_off_logits", None)
        self.off_stamps = getattr(speculator, "ngram_trace_off_stamps", None)
        if self.paired and (self.off_logits is None or self.off_stamps is None):
            raise RuntimeError("Paired trace requires on/off draft buffers")
        if self.logits is None or self.stamps is None:
            raise RuntimeError(
                "Ngram trace requires the instrumented DSpark speculator"
            )

    def record(
        self,
        batch,
        start,
        end,
        raw,
        processed,
        tokens,
        positions,
        sampled_counts,
        boundaries,
    ):
        if self.logits is None or self.stamps is None:
            raise RuntimeError("Draft trace buffers must be attached before recording")
        if raw.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Ngram trace must run outside CUDA graph capture")
        for local, request in enumerate(range(start, end)):
            rid = batch.req_ids[request]
            if self.prefix not in rid:
                continue
            slot = int(batch.idx_mapping_np[request])
            lo, hi = map(int, boundaries[local : local + 2])
            n = hi - lo - 1
            if n <= 0:
                continue
            anchor = int(positions[lo].item())
            if anchor < int(self.sampler.req_states.prefill_len.np[slot]):
                continue
            proposals = tokens[lo:hi].tolist()  # anchor, then draft tokens
            if any(t < 0 for t in proposals):
                continue
            stamps = self.stamps[slot, :n].tolist()
            if stamps != list(range(anchor + 1, anchor + n + 1)):
                raise RuntimeError(
                    f"Stale/misaligned draft logits for {rid}: "
                    f"{stamps}, anchor={anchor}"
                )
            if self.paired:
                if self.off_stamps is None or self.off_logits is None:
                    raise RuntimeError("Paired trace buffers must be attached")
                if self.off_stamps[slot, :n].tolist() != stamps:
                    raise RuntimeError(f"Stale/misaligned off logits for {rid}")
            accepted = int(sampled_counts[local].item()) - 1
            if not 0 <= accepted <= n:
                raise RuntimeError("Invalid accepted-prefix length")
            history = self.sampler.req_states.all_token_ids.gpu[
                slot, max(0, anchor - 2) : anchor
            ].tolist()
            temperature = float(self.sampler.sampling_states.temperature.np[slot])
            for step in range(n):
                if self.count >= self.limit:
                    raise RuntimeError(
                        "Ngram trace event limit reached; increase "
                        "VLLM_NGRAM_TRACE_MAX_EVENTS"
                    )
                history.append(int(proposals[step]))
                d = self.logits[slot, step].float()
                t = raw[lo + step].float()
                if d.numel() != t.numel():
                    raise RuntimeError(
                        "Ngram trace requires identical full target/draft vocabulary"
                    )
                p, q = t.softmax(-1), d.softmax(-1)
                residual = p - q
                chosen = residual.abs().topk(min(self.k, residual.numel())).indices
                proposal = int(proposals[step + 1])
                if temperature == 0:
                    actual_p = float(processed[lo + step].argmax().item() == proposal)
                    actual_q = 1.0
                else:
                    actual_p = float(
                        processed[lo + step].float().softmax(-1)[proposal].item()
                    )
                    actual_q = (
                        float((d / temperature).softmax(-1)[proposal].item())
                        if self.probabilistic
                        else 1.0
                    )
                fixed = []
                for order in (2, 3):
                    if len(history) < order:
                        continue
                    gram = tuple(history[-order:])
                    ids = self.candidates.get((gram, step))
                    if ids is not None:
                        fixed.append(
                            dict(
                                ngram_token_ids=gram,
                                token_ids=ids,
                                residual=residual[ids].tolist(),
                            )
                        )
                row = dict(
                    type="event",
                    request_id=rid,
                    anchor_position=anchor,
                    position=step,
                    token_position=anchor + step + 1,
                    ngram2=history[-2:] if len(history) >= 2 else None,
                    ngram3=history[-3:] if len(history) >= 3 else None,
                    proposed_token=proposal,
                    accepted_prefix=accepted,
                    reached=step <= accepted,
                    accepted=step < accepted,
                    first_rejection=step == accepted and accepted < n,
                    raw_l1=float(residual.abs().sum().item()),
                    raw_p_proposed=float(p[proposal].item()),
                    raw_q_proposed=float(q[proposal].item()),
                    verify_p_proposed=actual_p,
                    proposal_q=actual_q,
                    acceptance_probability=min(1.0, actual_p / max(actual_q, 1e-30)),
                    top_token_ids=chosen.tolist(),
                    top_residual=residual[chosen].tolist(),
                    fixed=fixed,
                    temperature=temperature,
                )
                if self.paired:
                    assert self.off_logits is not None
                    a_on = acceptance_overlap(
                        processed[lo + step], d, temperature, self.probabilistic
                    )
                    a_off = acceptance_overlap(
                        processed[lo + step],
                        self.off_logits[slot, step],
                        temperature,
                        self.probabilistic,
                    )
                    delta = a_on - a_off
                    row["contribution"] = dict(
                        a_on=a_on,
                        a_off=a_off,
                        delta_a=delta,
                        positive_gain=max(delta, 0.0),
                        negative_loss=max(-delta, 0.0),
                        oracle_a=max(a_on, a_off),
                    )
                self.file.write(json.dumps(row, allow_nan=False) + "\n")
                self.count += 1
