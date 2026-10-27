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


def pair(
    dtype=torch.float32,
    device="cpu",
    bank_size=13,
    max_length=128,
    num_speculative_tokens=7,
):
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
    return reference, LateConditioningStages(
        serving, 2, max_length, device, dtype, num_speculative_tokens
    )


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


@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("lengths", [[0, 0], [0, 73], [73, 0], [5, 31]])
@torch.inference_mode()
def test_graph_packed_capacity_matches_eager_without_padding_leak(bank_size, lengths):
    _, stages = pair(bank_size=bank_size)
    slots = [1, 0]
    context = torch.randn(sum(lengths), 32)
    stages.append_context_batch(slots, [0, 0], context, lengths)
    # Three graph rows but only two real requests; the last row is padding.
    # Run the exact graph body on CPU to check packing independently of CUDA.
    bank_graph = module.LateStageGraph(stages, 3, 128, False)
    bank_graph.slots[:2].copy_(torch.tensor(slots))
    bank_graph.lengths[:2].copy_(torch.tensor(lengths))
    starts = [n + 1 for n in lengths]
    bank_graph.starts[:2].copy_(torch.tensor(starts))
    expected_bank = stages.prepare_bank_batch(slots, lengths, starts)
    actual_bank = bank_graph.forward()
    torch.testing.assert_close(actual_bank[:2], expected_bank, atol=3e-6, rtol=3e-6)
    finish_graph = module.LateStageGraph(stages, 3, 128, True)
    finish_graph.slots.copy_(bank_graph.slots)
    finish_graph.lengths.copy_(bank_graph.lengths)
    finish_graph.starts.copy_(bank_graph.starts)
    finish_graph.banks.copy_(actual_bank)
    finish_graph.bonus.normal_()
    expected = stages.finish_batch(
        slots, lengths, expected_bank, starts, finish_graph.bonus[:2]
    )
    actual = finish_graph.forward()
    torch.testing.assert_close(actual[:2], expected, atol=3e-6, rtol=3e-6)
    # Unused rows must not affect active rows even with very different values.
    finish_graph.banks[2].fill_(1000)
    finish_graph.bonus[2].fill_(-1000)
    torch.testing.assert_close(finish_graph.forward()[:2], actual[:2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("proposal_length", [1, 6, 7])
@torch.inference_mode()
def test_stage_graph_replay_tracks_lengths_slots_and_preserves_outputs(
    bank_size, proposal_length
):
    _, stages = pair(
        torch.bfloat16, "cuda", bank_size, num_speculative_tokens=proposal_length
    )
    stages.append_context_batch(
        [0, 1],
        [0, 0],
        torch.randn(80, 32, device="cuda", dtype=torch.bfloat16),
        [40, 40],
    )
    stages.capture_graphs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        first = stages.prepare_bank_batch([1, 0], [5, 31], [7, 33])
        saved = first.clone()
        # Replaying the same graph must not mutate a previously returned bank.
        second = stages.prepare_bank_batch([0, 1], [32, 6], [33, 7])
    torch.cuda.current_stream().wait_stream(stream)
    first.record_stream(torch.cuda.current_stream())
    second.record_stream(torch.cuda.current_stream())
    saved.record_stream(torch.cuda.current_stream())
    torch.testing.assert_close(first, saved)
    for accepted in range(proposal_length + 1):
        lengths = [6 + accepted, 32]
        banks, starts = first, [7, 33]
        if bank_size == 12 and accepted == 7:
            banks = torch.cat((stages.prepare_bank_batch([1], [13], [14]), first[1:]))
            starts = [14, 33]
        bonus = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        actual = stages.finish_batch([1, 0], lengths, banks, starts, bonus)
        graphs, stages.graphs = stages.graphs, {}
        expected = stages.finish_batch([1, 0], lengths, banks, starts, bonus)
        stages.graphs = graphs
        torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("accepted", range(8))
@torch.inference_mode()
def test_graph_finish_preserves_all_acceptance_boundaries(bank_size, accepted):
    _, stages = pair(bank_size=bank_size)
    stages.append_context(1, 0, torch.randn(40, 32))
    length, start = 6 + accepted, 7
    bank = stages.prepare_bank_batch([1], [5], [start])
    if bank_size == 12 and accepted == 7:
        start = length + 1
        bank = stages.prepare_bank_batch([1], [length], [start])
    bonus = torch.randn(1, 16)
    expected = stages.finish_batch([1], [length], bank, [start], bonus)
    graph = module.LateStageGraph(stages, 2, 128, True)
    graph.slots[0] = 1
    graph.lengths[0] = length
    graph.starts[0] = start
    graph.banks[:1].copy_(bank)
    graph.bonus[:1].copy_(bonus)
    torch.testing.assert_close(graph.forward()[:1], expected, atol=3e-6, rtol=3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("width", [1, 6, 7, 12, 13])
@torch.inference_mode()
def test_graph_packer_writes_only_valid_kv_with_strided_values(width):
    from vllm.model_executor.models.dspark_late_kernels import pack_kv

    cache = torch.randn(2, 4, 640, 2, 8, device="cuda", dtype=torch.bfloat16)
    # Values are a strided view of QKV in production.
    qkv = torch.randn(3 * width, 48, device="cuda", dtype=torch.bfloat16)
    k, v = qkv[:, 16:32].reshape(-1, 2, 8), qkv[:, 32:].reshape(-1, 2, 8)
    slots = torch.tensor([3, 1, 0], device="cuda")
    lengths = torch.tensor([513, 64, 0], device="cuda")
    cu_k = torch.empty(4, device="cuda", dtype=torch.int32)
    out_k = torch.empty(3 * (2048 + width), 2, 8, device="cuda", dtype=torch.bfloat16)
    out_v = torch.empty_like(out_k)

    def metadata(values):
        lengths.copy_(torch.tensor(values, device="cuda"))
        cu_k[0] = 0
        cu_k[1:].copy_((lengths + width).cumsum(0))

    def pack():
        pack_kv(
            cache[0], cache[1], k, v, slots, lengths, cu_k, out_k, out_v, width, 2048
        )

    metadata([513, 64, 0])
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        pack()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        pack()
    stream.synchronize()
    for values in ([513, 64, 0], [0, 1, 0], [3, 0, 512]):
        metadata(values)
        out_k.fill_(123)
        out_v.fill_(-123)
        graph.replay()
        for output, source, query, sentinel in (
            (out_k, cache[0], k, 123),
            (out_v, cache[1], v, -123),
        ):
            expected = torch.cat(
                [
                    torch.cat(
                        (source[slot, :length], query[i * width : (i + 1) * width])
                    )
                    for i, (slot, length) in enumerate(zip([3, 1, 0], values))
                ]
            )
            torch.testing.assert_close(
                output[: expected.shape[0]], expected, atol=0, rtol=0
            )
            assert (output[expected.shape[0] :] == sentinel).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("bank_size", [12, 13])
@torch.inference_mode()
def test_bank_graph_matches_eager_across_capacity_buckets(bank_size):
    _, stages = pair(torch.bfloat16, "cuda", bank_size, max_length=640)
    stages.append_context_batch(
        [0, 1],
        [0, 0],
        torch.randn(1280, 32, device="cuda", dtype=torch.bfloat16),
        [640, 640],
    )
    cases = [([1, 0], [5, 31]), ([0, 1], [513, 64]), ([1], [129]), ([1, 0], [5, 31])]
    expected = [
        stages.prepare_bank_batch(slots, lengths, [n + 2 for n in lengths])
        for slots, lengths in cases
    ]
    stages.capture_graphs()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    actual = []
    with torch.cuda.stream(stream):
        for slots, lengths in cases:
            actual.append(
                stages.prepare_bank_batch(slots, lengths, [n + 2 for n in lengths])
            )
    # Target-like work on the main stream while bank graphs are queued.
    matrix = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    torch.mm(matrix, matrix)
    torch.cuda.current_stream().wait_stream(stream)
    for result, reference in zip(actual, expected):
        result.record_stream(torch.cuda.current_stream())
        torch.testing.assert_close(result, reference, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("bank_size", [12, 13])
@pytest.mark.parametrize("proposal_length", range(1, 8))
@pytest.mark.parametrize("all_accepted", [False, True])
@torch.inference_mode()
def test_variable_final_block_matches_reference_and_graph(
    bank_size, proposal_length, all_accepted
):
    from specforge.modeling.draft.dspark_late import finish_block

    ref, stages = pair(bank_size=bank_size, num_speculative_tokens=proposal_length)
    context = torch.randn(1, 24, 32)
    old, start = 5, 7
    anchor = old + 1 + (proposal_length if all_accepted else 0)
    stages.append_context(1, 0, context[0, :anchor])
    bank = stages.prepare_bank_batch([1], [old], [start])
    assert bank.shape == (1, bank_size, 16)
    # Proposal length must never change the bidirectional bank computation.
    stages.num_query_per_req = 7
    torch.testing.assert_close(bank, stages.prepare_bank_batch([1], [old], [start]))
    stages.num_query_per_req = proposal_length
    if bank_size == 12 and anchor - old > 7:
        start = anchor + 1
        bank = stages.prepare_bank_batch([1], [anchor], [start])
    bonus = torch.randn(1, 16)
    # Shorten only the reference final attention block, retaining the full bank.
    ref.block_size = proposal_length
    expected = finish_block(
        ref,
        bank[:, None],
        context,
        bonus[:, None],
        torch.tensor([[anchor]]),
        torch.tensor([[start]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    actual = stages.finish_batch([1], [anchor], bank, [start], bonus)
    assert actual.shape == (1, proposal_length, 16)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    graph = module.LateStageGraph(stages, 2, 128, True)
    graph.slots[0] = 1
    graph.lengths[0] = anchor
    graph.starts[0] = start
    graph.banks[:1].copy_(bank)
    graph.bonus[:1].copy_(bonus)
    torch.testing.assert_close(graph.forward()[:1], expected, atol=3e-6, rtol=3e-6)
