# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exactness of the SM80 manual e4m3fn encode/decode helpers.

Wrong rounding here drifts every KV-cache byte and surfaces as gradual
output corruption, so the encoder must match ``torch.Tensor.to(float8_e4m3fn)``
bit-for-bit on the clamped domain the kernels feed it, and the ALU decoder
must invert every byte exactly.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.fp8_sm80 import (
    _e4m3fn_to_f32_alu,
    _f32_to_e4m3fn_u8,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only kernels"
)


@triton.jit
def _encode_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, _f32_to_e4m3fn_u8(x), mask=mask)


@triton.jit
def _decode_kernel(u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    u = tl.load(u_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, _e4m3fn_to_f32_alu(u), mask=mask)


def _encode(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, dtype=torch.uint8)
    n = x.numel()
    _encode_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)
    return out


def _decode(u: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(u, dtype=torch.float32)
    n = u.numel()
    _decode_kernel[(triton.cdiv(n, 1024),)](u, out, n, BLOCK=1024)
    return out


@torch.inference_mode()
@pytest.mark.parametrize("draft_len", [3, 5])
def test_dspark_noncausal_fp8_decode_keeps_future_tokens(draft_len):
    """Exercise A100's draft index -> packed cache -> ragged attention path."""
    from tests.kernels.attention.test_rocm_triton_attn_dsv4 import (
        _pack_fp8_ds_mla_cache,
        _ref_sparse_decode_ragged,
    )
    from vllm.v1.attention.backends.mla.sparse_swa import (
        _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL,
    )
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        build_ragged_indices_from_dense,
        rocm_sparse_attn_decode,
    )

    device = "cuda"
    window, block_size, head_dim = 128, 64, 512
    seq_len = window + draft_len
    q = torch.zeros(draft_len + 1, 8, head_dim, dtype=torch.bfloat16, device=device)
    q[..., 0] = 1
    kv = torch.zeros(seq_len, head_dim, dtype=q.dtype, device=device)
    kv[-1, :2] = torch.tensor([128.0, 16.0], device=device)
    cache = _pack_fp8_ds_mla_cache(kv, block_size, use_fnuz=False)
    indices = torch.empty(draft_len + 1, 1, 256, dtype=torch.int32, device=device)
    lengths = torch.empty(draft_len + 1, dtype=torch.int32, device=device)
    valid = torch.arange(draft_len + 1, device=device) < draft_len
    _COMPUTE_DSPARK_NONCAUSAL_SWA_INDICES_KERNEL(
        indices,
        lengths,
        window,
        256,
        torch.tensor([0, draft_len], dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        torch.zeros(draft_len + 1, dtype=torch.int32, device=device),
        valid,
        torch.arange(3, dtype=torch.int32, device=device).reshape(1, 3),
        block_size,
        num_tokens=draft_len + 1,
        token_offset=0,
    )
    torch.testing.assert_close(lengths, valid.int() * seq_len)
    assert (indices[:draft_len, 0, seq_len - 1] == seq_len - 1).all()
    assert (indices[-1] == -1).all()
    ragged, indptr = build_ragged_indices_from_dense(indices[:, 0], lengths)
    output = torch.empty_like(q)
    rocm_sparse_attn_decode(
        q=q,
        kv_cache=None,
        swa_k_cache=cache,
        swa_only=True,
        topk_indices=None,
        topk_lens=None,
        swa_indices=indices,
        swa_lens=lengths,
        swa_ragged_indices=ragged,
        swa_ragged_indptr=indptr,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=None,
        scale=head_dim**-0.5,
        head_dim=head_dim,
        nope_head_dim=448,
        rope_head_dim=64,
        output=output,
    )
    expected = _ref_sparse_decode_ragged(
        q,
        cache,
        [list(range(seq_len))] * draft_len + [[]],
        head_dim**-0.5,
        None,
        block_size,
    )
    torch.testing.assert_close(output, expected, atol=2e-2, rtol=2e-2)
    # The first query's causal alternative excludes the only nonzero KV row.
    assert output[0, :, 1].min() > 1


def _assert_bytes_equal(got: torch.Tensor, ref: torch.Tensor, inputs: torch.Tensor):
    if not torch.equal(got, ref):
        bad = (got != ref).nonzero()[:10].flatten()
        raise AssertionError(
            f"{bad.numel()}+ mismatches, first at inputs "
            f"{inputs.flatten()[bad].tolist()}: got "
            f"{got.flatten()[bad].tolist()} ref {ref.flatten()[bad].tolist()}"
        )


def test_encoder_matches_torch_on_clamped_domain():
    """Kernels always clamp to +-448 before encoding; on that domain the
    manual encoder must equal torch's cast bitwise (RNE incl. ties)."""
    torch.manual_seed(0)
    xs = [
        (torch.rand(1_000_000, device="cuda") * 2 - 1) * 448.0,
        # log-uniform sweep into the subnormal range
        torch.exp2(torch.rand(1_000_000, device="cuda") * 30 - 21)
        * torch.where(torch.rand(1_000_000, device="cuda") > 0.5, 1.0, -1.0),
    ]
    x = torch.cat(xs).float()
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), ref, x)


def test_encoder_exact_ties_and_boundaries():
    vals = []
    # every representable value: exact roundtrip required
    all_bytes = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = all_bytes[(all_bytes & 0x7F) != 0x7F]
    vals.append(finite.view(torch.float8_e4m3fn).float())
    # exact midpoints between adjacent representables (RNE ties-to-even),
    # generated from consecutive finite positives
    pos = torch.sort(vals[0][vals[0] >= 0]).values
    mids = (pos[:-1] + pos[1:]) / 2
    vals += [mids, -mids]
    # boundary/overflow: values at and beyond max finite saturate to +-448
    vals.append(
        torch.tensor(
            [448.0, 449.0, 455.9, 456.0, 464.0, 1e30, -1e30, 0.0, -0.0],
            device="cuda",
        )
    )
    # deep subnormals and half-min-subnormal ties
    vals.append(
        torch.tensor(
            [2**-9, 2**-10, 1.5 * 2**-10, 2**-11, 2**-20, -(2**-10)],
            device="cuda",
        )
    )
    x = torch.cat([v.float().flatten() for v in vals])
    ref = x.to(torch.float8_e4m3fn).view(torch.uint8)
    _assert_bytes_equal(_encode(x), ref, x)


def test_alu_decoder_all_bytes():
    """ALU decode must match torch's fp8->f32 for all 256 bytes, incl. the
    two NaN encodings (NaN in, NaN out -- keeps NaN scrubbing identical)."""
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    got = _decode(u)
    ref = u.view(torch.float8_e4m3fn).float()
    assert torch.equal(got.nan_to_num(1234.5), ref.nan_to_num(1234.5))
    assert got[(u & 0x7F) == 0x7F].isnan().all()


def test_roundtrip_all_finite_bytes():
    u = torch.arange(256, dtype=torch.uint8, device="cuda")
    finite = u[(u & 0x7F) != 0x7F]
    assert torch.equal(_encode(_decode(finite)), finite)
