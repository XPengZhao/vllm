# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-buffer diagnostic copies captured alongside DSpark drafting."""

from vllm.triton_utils import tl, triton


@triton.jit
def _copy(
    logits,
    mapping,
    positions,
    cache,
    stamps,
    stride,
    map_stride,
    pos_stride,
    vocab: tl.constexpr,
    steps: tl.constexpr,
    step: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    slot = tl.load(mapping + row * map_stride).to(tl.int64)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * stride + offsets, offsets < vocab, 0)
    tl.store(
        cache + (slot * steps + step) * vocab + offsets,
        values,
        (slot >= 0) & (offsets < vocab),
    )
    if block == 0:
        pos = tl.load(positions + row * pos_stride)
        tl.store(stamps + slot * steps + step, pos, slot >= 0)


def cache_trace_logits(logits, mapping, positions, cache, stamps, step):
    _copy[(logits.shape[0], triton.cdiv(logits.shape[1], 1024))](
        logits,
        mapping,
        positions,
        cache,
        stamps,
        logits.stride(0),
        mapping.stride(0),
        positions.stride(0),
        logits.shape[1],
        cache.shape[1],
        step,
        1024,
    )
