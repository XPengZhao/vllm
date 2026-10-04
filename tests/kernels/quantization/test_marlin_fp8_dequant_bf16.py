# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MARLIN_FP8_DEQUANT_BF16: load-time dequant of block-fp8 Marlin
layers to the model dtype (cuBLAS route).

CPU-only on purpose: the dequant path is plain torch, and the end-to-end
flag-on server A/B is what validates it on GPU.
"""

from unittest.mock import Mock

import pytest
import torch

import vllm.model_executor.kernels.linear.scaled_mm.marlin as marlin
from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
    MarlinFP8ScaledMMLinearKernel,
)

QB = 128


def _make_layer(
    n: int, k: int, scale_dtype: torch.dtype, scale_name: str = "weight_scale_inv"
) -> torch.nn.Module:
    torch.manual_seed(0)
    n_blocks, k_blocks = (n + QB - 1) // QB, (k + QB - 1) // QB
    wv = torch.randn(n_blocks, QB, k_blocks, QB, dtype=torch.float32) * 0.02
    amax = wv.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-12)
    if scale_dtype == torch.float8_e8m0fnu:
        # ue8m0 scales are power-of-two; quantize amax/448 up to one.
        scales = (amax / 448.0).log2().ceil().exp2()
    else:
        scales = amax / 448.0
    w_fp8 = (
        (wv / scales)
        .clamp(-448.0, 448.0)
        .view(n_blocks * QB, k_blocks * QB)[:n, :k]
        .contiguous()
        .to(torch.float8_e4m3fn)
    )

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(w_fp8, requires_grad=False)
    layer.register_parameter(
        scale_name,
        torch.nn.Parameter(
            scales.view(n_blocks, k_blocks).to(scale_dtype), requires_grad=False
        ),
    )
    layer.orig_dtype = torch.bfloat16
    layer.weight_block_size = [QB, QB]
    layer.output_size_per_partition = n
    layer.input_size_per_partition = k
    layer.prefix = "model.layers.0.self_attn.q_b_proj"
    return layer


@pytest.fixture
def kernel(monkeypatch):
    monkeypatch.setenv("VLLM_MARLIN_FP8_DEQUANT_BF16", "1")
    monkeypatch.delenv("VLLM_MARLIN_FP8_DEQUANT_EXCLUDE", raising=False)
    monkeypatch.setenv("VLLM_ROCM_FP8_PADDING", "0")
    monkeypatch.setattr(marlin.current_platform, "is_fp8_fnuz", lambda: False)
    # Bypass only the hardware-gated constructor; exercise public postload/apply.
    kernel = object.__new__(MarlinFP8ScaledMMLinearKernel)
    kernel.block_quant = True
    kernel.size_k_first = False
    kernel.marlin_input_dtype = None
    return kernel


@pytest.mark.parametrize("n,k", [(256, 384), (130, 260)])
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
@pytest.mark.parametrize("scale_name", ["weight_scale_inv", "weight_scale"])
def test_postload_dequant_matches_reference_and_frees_fp8(
    kernel, monkeypatch, n, k, scale_dtype, scale_name
):
    layer = _make_layer(n, k, scale_dtype, scale_name)
    scale_full = (
        getattr(layer, scale_name)
        .to(torch.float32)
        .repeat_interleave(QB, 0)
        .repeat_interleave(QB, 1)[:n, :k]
    )
    ref = (layer.weight.to(torch.float32) * scale_full).to(torch.bfloat16)
    prepare = Mock(side_effect=AssertionError("Dequant must precede Marlin repack"))
    monkeypatch.setattr(marlin, "prepare_fp8_layer_for_marlin", prepare)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.shape == (n, k)
    torch.testing.assert_close(layer.weight, ref, rtol=0, atol=0)
    assert not hasattr(layer, scale_name)
    assert layer.marlin_fp8_dequant
    prepare.assert_not_called()

    x = torch.randn(2, 4, k, dtype=torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(out, torch.nn.functional.linear(x, ref, bias))


@pytest.mark.parametrize("reason", ["disabled", "excluded", "non_block"])
def test_ineligible_layers_keep_marlin_execution(kernel, monkeypatch, reason):
    layer = _make_layer(QB, QB, torch.float32, "weight_scale")
    if reason == "disabled":
        monkeypatch.setenv("VLLM_MARLIN_FP8_DEQUANT_BF16", "0")
    elif reason == "excluded":
        monkeypatch.setenv("VLLM_MARLIN_FP8_DEQUANT_EXCLUDE", " mlp, q_b_proj ")
    else:
        kernel.block_quant = False
        kernel.size_k_first = True

    prepare = Mock()
    apply = Mock(return_value=torch.zeros(2, QB, dtype=torch.bfloat16))
    monkeypatch.setattr(marlin, "prepare_fp8_layer_for_marlin", prepare)
    monkeypatch.setattr(marlin, "apply_fp8_marlin_linear", apply)
    layer.workspace = torch.empty(0)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert hasattr(layer, "weight_scale")
    assert not getattr(layer, "marlin_fp8_dequant", False)
    prepare.assert_called_once_with(
        layer, kernel.size_k_first, input_dtype=kernel.marlin_input_dtype
    )
    x = torch.randn(2, QB, dtype=torch.bfloat16)
    bias = torch.randn(QB, dtype=torch.bfloat16)
    assert kernel.apply_weights(layer, x, bias) is apply.return_value
    apply.assert_called_once_with(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        workspace=layer.workspace,
        size_n=QB,
        size_k=QB,
        input_dtype=None,
        bias=bias,
    )


def test_bmm_weights_stay_raw_fp8_when_dequant_enabled(kernel, monkeypatch):
    layer = _make_layer(QB, QB, torch.float8_e8m0fnu)
    layer.is_bmm = True
    weight = layer.weight.to(torch.float32)
    scale = layer.weight_scale_inv.to(torch.float32)
    prepare = Mock(side_effect=AssertionError("BMM weights must not be repacked"))
    monkeypatch.setattr(marlin, "prepare_fp8_layer_for_marlin", prepare)

    kernel.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale_inv.dtype == torch.float8_e8m0fnu
    assert not getattr(layer, "marlin_fp8_dequant", False)
    torch.testing.assert_close(layer.weight.to(torch.float32), weight)
    torch.testing.assert_close(layer.weight_scale_inv.to(torch.float32), scale)
    prepare.assert_not_called()
