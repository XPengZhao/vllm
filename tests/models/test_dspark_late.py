# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare native serving stages to the 13-state training implementation.

Set PYTHONPATH to the matching SpecForge checkout. No vLLM CUDA extension is
needed for the CPU stage comparisons; the CUDA overlap test is optional.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

SOURCE = (
    Path(__file__).resolve().parents[2] / "vllm/model_executor/models/dspark_late.py"
)
spec = importlib.util.spec_from_file_location("late_stages", SOURCE)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
LateConditioningStages = module.LateConditioningStages


def pair(dtype=torch.float32, device="cpu", bank_size=13):
    reference_module = pytest.importorskip("specforge.modeling.draft.dspark")
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    config.block_size = 7
    config.num_target_layers = 2
    config.dflash_config = dict(
        projector_type="dspark",
        target_layer_ids=[0, 1],
        markov_rank=4,
        late_conditioning=dict(enabled=True, bank_size=13, max_context_lag=8),
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(42)
    reference = reference_module.DSparkDraftModel(config).to(device, dtype).eval()

    def norm(value):
        return SimpleNamespace(
            weight=value.weight, variance_epsilon=value.variance_epsilon
        )

    layers = []
    for layer in reference.layers:
        attn = layer.self_attn
        qkv = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight])
        gate_up = torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight])
        layers.append(
            SimpleNamespace(
                input_layernorm=norm(layer.input_layernorm),
                post_attention_layernorm=norm(layer.post_attention_layernorm),
                self_attn=SimpleNamespace(
                    qkv_proj=SimpleNamespace(weight=qkv),
                    q_size=attn.q_proj.weight.shape[0],
                    kv_size=attn.k_proj.weight.shape[0],
                    q_norm=norm(attn.q_norm),
                    k_norm=norm(attn.k_norm),
                    o_proj=attn.o_proj,
                ),
                mlp=SimpleNamespace(
                    gate_up_proj=SimpleNamespace(weight=gate_up),
                    down_proj=layer.mlp.down_proj,
                ),
            )
        )
    serving = SimpleNamespace(
        config=config,
        model=SimpleNamespace(
            layers=layers,
            fc=reference.fc,
            hidden_norm=norm(reference.hidden_norm),
            norm=norm(reference.norm),
            future_mask=reference.future_mask,
            anchor_adapter=reference.anchor_adapter,
        ),
    )
    # Legacy weights have identical parameter shapes; their attention bank
    # width and training boundary distribution differ.
    config.dflash_config["late_conditioning"] = dict(
        enabled=True, bank_size=bank_size, max_context_lag=bank_size - 5
    )
    return reference, LateConditioningStages(serving, 2, 128, device, dtype)


def legacy_training_forward(ref, context, bonus, anchor, old):
    """The bank12/lag0..7 training computation from SpecForge f1e4ba4."""
    from specforge.modeling.draft.dspark_late import _stage, finish_block

    start = old + (1 if old == anchor else 2)
    keep = torch.ones(1, 1, dtype=torch.bool, device=context.device)
    starts = torch.tensor([[start]], device=context.device)
    positions = starts[..., None] + torch.arange(12, device=context.device)
    states = ref.future_mask.view(1, 1, 1, -1).expand(1, 1, 12, -1)
    bank = _stage(
        ref,
        states,
        ref.hidden_norm(ref.fc(context)),
        positions,
        torch.tensor([[old]], device=context.device),
        keep,
        ref.layers[:-1],
    )
    hidden = finish_block(
        ref,
        bank,
        context,
        bonus,
        torch.tensor([[anchor]], device=context.device),
        starts,
        keep,
    )
    return bank[0, 0], hidden[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lag", range(8))
@torch.inference_mode()
def test_legacy_stages_match_training_without_widening_bank(dtype, lag):
    ref, stages = pair(dtype, bank_size=12)
    context = torch.randn(1, 24, 32, dtype=dtype)
    anchor, old = 13, 13 - lag
    start = old + (1 if lag == 0 else 2)
    stages.append_context(0, 0, context[0, :old])
    bank = stages.prepare_bank(0, old, start)
    stages.append_context(0, old, context[0, old:anchor])
    bonus = torch.randn(1, 1, 16, dtype=dtype)
    hidden = stages.finish(0, anchor, bank, start, bonus)
    expected_bank, expected = legacy_training_forward(ref, context, bonus, anchor, old)
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-6
    assert bank.shape == (12, 16)
    assert hidden.shape == (7, 16)
    torch.testing.assert_close(bank, expected_bank, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(hidden, expected, atol=tolerance, rtol=tolerance)
    with pytest.raises(ValueError, match="boundary"):
        stages.finish(0, start + 6, bank, start, bonus)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("accepted", range(8))
@torch.inference_mode()
def test_serving_stages_match_training_at_every_boundary(dtype, accepted):
    from specforge.modeling.draft.dspark_late import finish_block, prepare_bank

    ref, stages = pair(dtype)
    context = torch.randn(1, 24, 32, dtype=dtype)
    old = 5
    anchor = old + accepted + 1
    keep = torch.ones(1, 1, dtype=torch.bool)
    stages.append_context(0, 0, context[0, :old])
    bank = stages.prepare_bank(0, old, old + 2)
    expected_bank = prepare_bank(
        ref, context, torch.tensor([[old]]), torch.tensor([[old + 2]]), keep
    )
    tolerance = 3e-2 if dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(
        bank, expected_bank[0, 0], atol=tolerance, rtol=tolerance
    )
    stages.append_context(0, old, context[0, old:anchor])
    bonus = torch.randn(1, 1, 16, dtype=dtype)
    hidden = stages.finish(0, anchor, bank, old + 2, bonus)
    expected = finish_block(
        ref,
        expected_bank,
        context,
        bonus,
        torch.tensor([[anchor]]),
        torch.tensor([[old + 2]]),
        keep,
    )
    torch.testing.assert_close(hidden, expected[0], atol=tolerance, rtol=tolerance)
    # Real labels cover seven predictions; the all-accepted boundary must not
    # require an eighth final-layer input or an out-of-range bank selection.
    assert hidden.shape == (7, 16)


@torch.inference_mode()
def test_bootstrap_and_request_slots_are_isolated():
    from specforge.modeling.draft.dspark_late import training_forward

    ref, stages = pair()
    context = torch.randn(1, 20, 32)
    stages.append_context(0, 0, context[0, :9])
    stages.append_context(1, 0, torch.randn(20, 32) * 100)
    bank = stages.prepare_bank(0, 9, 10)
    bonus = torch.randn(1, 1, 16)
    actual = stages.finish(0, 9, bank, 10, bonus)
    expected = training_forward(
        ref,
        context,
        bonus,
        torch.tensor([[9]]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.tensor([[9]]),
    )
    torch.testing.assert_close(actual, expected[0], atol=3e-6, rtol=3e-6)
    with pytest.raises(ValueError, match="boundary"):
        stages.finish(0, 19, bank, 10, bonus)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_cuda_background_bank_matches_serial_after_context_update():
    _, stages = pair(torch.bfloat16, "cuda")
    context = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    stages.append_context(0, 0, context[:5])
    serial = stages.prepare_bank(0, 5, 7)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        concurrent = stages.prepare_bank(0, 5, 7)
        event = torch.cuda.Event()
        event.record()
    # Independent verifier work on the main stream, while the bank is in flight.
    target = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    torch.mm(target, target)
    torch.cuda.current_stream().wait_event(event)
    concurrent.record_stream(torch.cuda.current_stream())
    stages.append_context(0, 5, context[5:13])
    torch.testing.assert_close(concurrent, serial)
    bonus = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        stages.finish(0, 13, concurrent, 7, bonus),
        stages.finish(0, 13, serial, 7, bonus),
    )


@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_packed_batch_matches_independent_training_with_unequal_contexts(
    bank_size, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from specforge.modeling.draft.dspark_late import training_forward

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    ref, stages = pair(dtype, device, bank_size)
    lengths, slots = [0, 73], [1, 0]
    contexts = [torch.randn(n, 32, dtype=dtype, device=device) for n in lengths]
    stages.append_context_batch(slots, [0, 0], torch.cat(contexts), lengths)
    cached = stages.kv.clone()
    starts = [n + 1 for n in lengths]
    banks = stages.prepare_bank_batch(slots, lengths, starts)
    bonuses = torch.randn(2, 16, dtype=dtype, device=device)
    actual = stages.finish_batch(slots, lengths, banks, starts, bonuses)
    for i, n in enumerate(lengths):
        # Independent training reference catches cross-request attention and
        # packed offset errors, including an empty committed prefix.
        context = torch.cat((contexts[i], contexts[i].new_zeros(1, 32)))[None]
        if bank_size == 12:
            _, expected = legacy_training_forward(
                ref, context, bonuses[i].reshape(1, 1, 16), n, n
            )
        else:
            expected = training_forward(
                ref,
                context,
                bonuses[i].reshape(1, 1, 16),
                torch.tensor([[n]], device=device),
                torch.ones(1, 1, dtype=torch.bool, device=device),
                torch.tensor([[n]], device=device),
            )[0]
        tolerance = 3e-2 if device == "cuda" else 3e-6
        torch.testing.assert_close(actual[i], expected, atol=tolerance, rtol=tolerance)
    # Bank/self attention must never persist speculative K/V.
    for slot, n in zip(slots, lengths):
        torch.testing.assert_close(stages.kv[:, :, slot, :n], cached[:, :, slot, :n])
