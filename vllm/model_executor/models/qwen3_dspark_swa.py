# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Anchor-window attention for the SpecForge Qwen3 DSpark MLA ablation."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _anchor_attention(
    Q,
    K,
    V,
    Out,
    Cache,
    Table,
    Starts,
    Positions,
    table_stride,
    cache_page_stride,
    cache_token_stride,
    cache_head_stride,
    PAGE: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    SCALE: tl.constexpr,
):
    req, head = tl.program_id(0), tl.program_id(1)
    start = tl.load(Starts + req)
    size = tl.load(Starts + req + 1) - start
    if size == 0:
        return
    anchor = tl.load(Positions + start)
    m, d = tl.arange(0, M), tl.arange(0, DIM)
    row = ((start + m[:, None]) * HEADS + head) * DIM + d[None, :]
    q = tl.load(Q + row, m[:, None] < size, other=0)
    # Include the entire live block first, including positions after the query.
    k = tl.load(K + row, m[:, None] < size, other=0)
    v = tl.load(V + row, m[:, None] < size, other=0)
    scores = tl.dot(q, tl.trans(k)) * SCALE
    scores = tl.where(m[None, :] < size, scores, -float("inf"))
    peak = tl.max(scores, axis=1)
    probs = tl.exp(scores - peak[:, None])
    total = tl.sum(probs, axis=1)
    acc = tl.dot(probs.to(v.dtype), v)

    first = tl.maximum(anchor - WINDOW + 1, 0)
    for tile in range(tl.cdiv(anchor - first, N)):
        pos = first + tile * N + tl.arange(0, N)
        valid = pos < anchor
        page = tl.load(Table + req * table_stride + pos // PAGE, valid, other=0)
        offset = (
            page.to(tl.int64) * cache_page_stride
            + (pos % PAGE) * cache_token_stride
            + head * cache_head_stride
        )
        addr = offset[:, None] + d[None, :]
        history_k = tl.load(Cache + addr, valid[:, None], other=0)
        history_v = tl.load(Cache + addr + DIM, valid[:, None], other=0)
        history_scores = tl.dot(q, tl.trans(history_k)) * SCALE
        history_scores = tl.where(valid[None, :], history_scores, -float("inf"))
        new_peak = tl.maximum(peak, tl.max(history_scores, axis=1))
        alpha = tl.exp(peak - new_peak)
        history_probs = tl.exp(history_scores - new_peak[:, None])
        acc = acc * alpha[:, None] + tl.dot(
            history_probs.to(history_v.dtype), history_v
        )
        total = total * alpha + tl.sum(history_probs, axis=1)
        peak = new_peak
    tl.store(Out + row, acc / total[:, None], m[:, None] < size)


def anchor_window_attention(query, key, value, cache, metadata, positions, window):
    # FlashAttention exposes [page, head, token, K|V], possibly with HND strides.
    pages = cache.transpose(1, 2)
    heads, dim = key.shape[1:]
    assert pages.shape[-2:] == (heads, 2 * dim) and pages.stride(-1) == 1
    # Graph padding has no live request. Leave those output rows zero.
    out = torch.zeros_like(query)
    _anchor_attention[(metadata.query_start_loc.numel() - 1, heads)](
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        out,
        pages,
        metadata.block_table,
        metadata.query_start_loc,
        positions,
        metadata.block_table.stride(0),
        *pages.stride()[:3],
        PAGE=pages.shape[1],
        HEADS=heads,
        DIM=dim,
        WINDOW=window,
        M=max(16, triton.next_power_of_2(metadata.max_query_len)),
        N=32,
        SCALE=dim**-0.5,
    )
    return out
