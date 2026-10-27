# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe packing of committed context and temporary DSpark self K/V."""

from vllm.triton_utils import tl, triton


@triton.jit
def _pack_kv(
    cache_k,
    cache_v,
    self_k,
    self_v,
    slots,
    lengths,
    cu_k,
    out_k,
    out_v,
    CACHE_SLOT_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    K_TOKEN_STRIDE: tl.constexpr,
    V_TOKEN_STRIDE: tl.constexpr,
    HEAD_WIDTH: tl.constexpr,
    QUERY_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    length = tl.load(lengths + req)
    # Bounded launch grid; loop only over real context and self K/V.
    for tile in range(
        tl.program_id(1),
        tl.cdiv((length + QUERY_WIDTH) * HEAD_WIDTH, BLOCK),
        tl.num_programs(1),
    ):
        flat = tile * BLOCK + tl.arange(0, BLOCK)
        token, channel = flat // HEAD_WIDTH, flat % HEAD_WIDTH
        valid = token < length + QUERY_WIDTH
        context = token < length
        slot = tl.load(slots + req)
        cache_offset = slot * CACHE_SLOT_STRIDE + token * CACHE_TOKEN_STRIDE + channel
        k = tl.load(cache_k + cache_offset, mask=valid & context, other=0)
        v = tl.load(cache_v + cache_offset, mask=valid & context, other=0)
        query = req * QUERY_WIDTH + token - length
        qk = tl.load(
            self_k + query * K_TOKEN_STRIDE + channel, mask=valid & ~context, other=0
        )
        qv = tl.load(
            self_v + query * V_TOKEN_STRIDE + channel, mask=valid & ~context, other=0
        )
        destination = (tl.load(cu_k + req) + token) * HEAD_WIDTH + channel
        tl.store(out_k + destination, tl.where(context, k, qk), mask=valid)
        tl.store(out_v + destination, tl.where(context, v, qv), mask=valid)


def pack_kv(
    cache_k, cache_v, k, v, slots, lengths, cu_k, out_k, out_v, width, capacity
):
    head_width = k.shape[1] * k.shape[2]
    _pack_kv[
        (slots.numel(), min(128, triton.cdiv((capacity + width) * head_width, 1024)))
    ](
        cache_k,
        cache_v,
        k,
        v,
        slots,
        lengths,
        cu_k,
        out_k,
        out_v,
        cache_k.stride(0),
        cache_k.stride(1),
        k.stride(0),
        v.stride(0),
        head_width,
        width,
        1024,
    )
