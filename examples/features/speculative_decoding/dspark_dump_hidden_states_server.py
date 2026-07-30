# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dump DeepSeek-V4 DSpark hidden states from a running vLLM server.

The server should run the normal DSpark speculative path. This script sends one
pre-tokenized request at a time and asks the model runner to dump only the
response-context hidden-state chunks, then merges them into the SpecForge
offline-training format.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


ASSISTANT_MARKER = "<｜Assistant｜>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--tmp-path", default=None)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--response-context-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            rows.append(row)
    return rows


def get_text(row: dict[str, Any]) -> str:
    text = row.get("text")
    if isinstance(text, str):
        return text
    prompt = row.get("prompt")
    if isinstance(prompt, str):
        return prompt
    raise ValueError("Each row must contain a string 'text' or 'prompt' field")


def tokenize(tokenizer, text: str) -> tuple[list[int], int, int]:
    marker_pos = text.rfind(ASSISTANT_MARKER)
    if marker_pos < 0:
        raise ValueError(f"Missing assistant marker: {ASSISTANT_MARKER}")
    response_text_start = marker_pos + len(ASSISTANT_MARKER)
    if not text[response_text_start:].strip():
        raise ValueError("Empty assistant response")

    prefix = text[:response_text_start]
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    response_start = len(tokenizer.encode(prefix, add_special_tokens=False))
    response_end = len(input_ids)
    if response_start >= response_end:
        raise ValueError("No response tokens after assistant marker")
    return input_ids, response_start, response_end


def truncate_left(
    input_ids: list[int],
    response_start: int,
    response_end: int,
    max_length: int,
) -> tuple[list[int], int, int]:
    if len(input_ids) <= max_length:
        return input_ids, response_start, response_end
    dropped = len(input_ids) - max_length
    input_ids = input_ids[dropped:]
    response_start = max(response_start - dropped, 0)
    response_end = response_end - dropped
    if response_start >= response_end:
        raise ValueError("Response was truncated away")
    return input_ids, response_start, response_end


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def rows_dir_for(output_root: Path, index: int, group_size: int = 2000) -> Path:
    start = index // group_size * group_size
    end = start + group_size
    return output_root / f"rows_{start}-{end}"


def build_opd_features(
    row: dict[str, Any],
    *,
    response_start: int,
    save_start: int,
) -> dict[str, torch.Tensor]:
    target_rollout = row.get("target_rollout")
    spec_decode = (
        target_rollout.get("spec_decode")
        if isinstance(target_rollout, dict)
        else None
    )
    trace = spec_decode.get("trace") if isinstance(spec_decode, dict) else None
    if trace is None:
        return {}
    if not isinstance(trace, list):
        raise ValueError("target_rollout.spec_decode.trace must be a list")

    records: list[tuple[int, list[int], list[float], int]] = []
    for entry in trace:
        if not isinstance(entry, dict):
            raise ValueError("each speculative trace entry must be an object")
        prefix_length = entry.get("response_prefix_length")
        draft_token_ids = entry.get("draft_token_ids")
        target_logprobs = entry.get("target_logprobs")
        accepted_length = entry.get("accepted_length")
        if (
            not isinstance(prefix_length, int)
            or not isinstance(draft_token_ids, list)
            or not isinstance(target_logprobs, list)
            or not isinstance(accepted_length, int)
            or len(draft_token_ids) != len(target_logprobs)
        ):
            raise ValueError(f"invalid speculative trace entry: {entry}")
        anchor = response_start + prefix_length - 1 - save_start
        if anchor < 0:
            raise ValueError(f"speculative anchor precedes saved context: {anchor}")
        records.append(
            (
                anchor,
                [int(token_id) for token_id in draft_token_ids],
                [float(logprob) for logprob in target_logprobs],
                accepted_length,
            )
        )

    max_candidates = max((len(record[1]) for record in records), default=0)
    num_blocks = len(records)
    anchor_positions = torch.zeros(num_blocks, dtype=torch.long)
    draft_token_ids = torch.zeros(
        num_blocks, max_candidates, dtype=torch.long
    )
    target_logprobs = torch.zeros(num_blocks, max_candidates, dtype=torch.float32)
    accepted_lengths = torch.zeros(num_blocks, dtype=torch.long)
    candidate_mask = torch.zeros(num_blocks, max_candidates, dtype=torch.bool)
    for index, (anchor, token_ids, logprobs, accepted_length) in enumerate(records):
        count = len(token_ids)
        anchor_positions[index] = anchor
        accepted_lengths[index] = min(accepted_length, count)
        if count:
            draft_token_ids[index, :count] = torch.tensor(token_ids)
            target_logprobs[index, :count] = torch.tensor(logprobs)
            candidate_mask[index, :count] = True
    return {
        "opd_anchor_positions": anchor_positions,
        "opd_draft_token_ids": draft_token_ids,
        "opd_target_logprobs": target_logprobs,
        "opd_accepted_lengths": accepted_lengths,
        "opd_candidate_mask": candidate_mask,
    }


def merge_direct_parts(
    prefix: Path,
    dst: Path,
    *,
    keep_parts: bool,
    save_start: int,
    save_end: int,
    response_start: int,
    row: dict[str, Any],
) -> None:
    parts = sorted(
        prefix.parent.glob(prefix.name + ".part_*_*.pt"),
        key=lambda p: int(p.name.rsplit(".part_", 1)[1].split("_", 1)[0]),
    )
    if not parts:
        raise FileNotFoundError(f"No direct hidden-state parts found for {prefix}")

    chunks = [torch.load(part, map_location="cpu", weights_only=False) for part in parts]
    chunks = sorted(chunks, key=lambda item: int(item["start"]))
    for prev, cur in zip(chunks, chunks[1:]):
        if int(prev["end"]) != int(cur["start"]):
            raise ValueError(
                "Non-contiguous direct hidden-state parts: "
                f"{prev['end']} -> {cur['start']}"
            )

    input_ids = torch.cat([item["input_ids"] for item in chunks], dim=0)
    loss_mask = torch.cat([item["loss_mask"] for item in chunks], dim=0)
    hidden_state = torch.cat(
        [item["hidden_state"] for item in chunks], dim=0
    ).unsqueeze(0)
    aux_hidden_state = torch.cat(
        [item["aux_hidden_state"] for item in chunks], dim=0
    ).unsqueeze(0)

    if int(chunks[0]["start"]) != save_start or int(chunks[-1]["end"]) != save_end:
        raise ValueError(
            "Direct hidden-state parts do not cover requested span: "
            f"{chunks[0]['start']}..{chunks[-1]['end']} vs {save_start}..{save_end}"
        )
    if input_ids.numel() != save_end - save_start:
        raise ValueError(
            f"Token count mismatch: {input_ids.numel()} vs {save_end - save_start}"
        )
    if hidden_state.shape[1] != input_ids.numel():
        raise ValueError(
            f"hidden_state length mismatch: {hidden_state.shape} vs {input_ids.numel()}"
        )
    if aux_hidden_state.shape[1] != input_ids.numel():
        raise ValueError(
            "aux_hidden_state length mismatch: "
            f"{aux_hidden_state.shape} vs {input_ids.numel()}"
        )
    if hidden_state.shape[-1] != 4096:
        raise ValueError(f"Expected final hidden size 4096, got {hidden_state.shape}")
    if aux_hidden_state.shape[-1] != 12288:
        raise ValueError(f"Expected aux hidden size 12288, got {aux_hidden_state.shape}")
    expected_loss_tokens = max(save_end - max(response_start, save_start), 0)
    if int(loss_mask.sum().item()) != expected_loss_tokens:
        raise ValueError(
            "loss_mask token count mismatch: "
            f"{int(loss_mask.sum().item())} vs {expected_loss_tokens}"
        )

    sample = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "hidden_state": hidden_state,
        "aux_hidden_state": aux_hidden_state,
        "metadata": {
            "save_start": save_start,
            "save_end": save_end,
            "response_start": response_start,
        },
    }
    sample.update(
        build_opd_features(
            row,
            response_start=response_start,
            save_start=save_start,
        )
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sample, dst)
    if not keep_parts:
        for part in parts:
            part.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_root = Path(args.output_path)
    tmp_root = Path(args.tmp_path or output_root / "_vllm_parts")
    tmp_root.mkdir(parents=True, exist_ok=True)
    if args.max_length <= 1:
        raise ValueError("--max-length must be greater than 1")
    max_prompt_length = args.max_length - 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    rows = read_jsonl(data_path)
    if args.limit is not None:
        rows = rows[args.start_index : args.start_index + args.limit]
    else:
        rows = rows[args.start_index :]

    endpoint = args.server_url.rstrip("/") + "/v1/completions"
    for local_idx, row in enumerate(tqdm(rows, desc="Dumping DSpark hidden states")):
        global_idx = args.start_index + local_idx
        rows_dir = rows_dir_for(output_root, global_idx)
        rows_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = rows_dir / f"data_{global_idx}.ckpt"
        if ckpt_path.exists():
            continue

        text = get_text(row)
        input_ids, response_start, response_end = tokenize(tokenizer, text)
        input_ids, response_start, response_end = truncate_left(
            input_ids,
            response_start,
            response_end,
            max_prompt_length,
        )

        save_start = max(response_start - args.response_context_tokens, 0)
        save_end = response_end
        part_prefix = tmp_root / f"data_{global_idx}"
        part_prefix.parent.mkdir(parents=True, exist_ok=True)
        for stale_part in part_prefix.parent.glob(part_prefix.name + ".part_*_*.pt"):
            stale_part.unlink()
        payload = {
            "prompt": input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "kv_transfer_params": {
                "direct_hidden_states_path": str(part_prefix),
                "save_start_token": save_start,
                "save_end_token": save_end,
                "loss_start_token": response_start,
            },
        }
        if args.model is not None:
            payload["model"] = args.model

        post_json(endpoint, payload, args.request_timeout)
        merge_direct_parts(
            part_prefix,
            ckpt_path,
            keep_parts=args.keep_parts,
            save_start=save_start,
            save_end=save_end,
            response_start=response_start,
            row=row,
        )


if __name__ == "__main__":
    main()
