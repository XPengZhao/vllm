# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA engine acceptance smoke with real exported tiny target/draft weights."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("draft_sample_method", ["greedy", "probabilistic"])
@pytest.mark.parametrize("legacy", [False, True])
def test_late_engine_matches_target_and_reuses_request_slots(
    tmp_path, vllm_runner, draft_sample_method, legacy, monkeypatch
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from vllm import SamplingParams

    draft_module = pytest.importorskip("specforge.modeling.draft.dspark")
    torch.manual_seed(42)
    target_config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=63,
        pad_token_id=0,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
    )
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    Qwen3ForCausalLM(target_config).save_pretrained(target_dir)
    config = Qwen3Config.from_dict(target_config.to_dict())
    config.num_hidden_layers = 5
    config.layer_types = ["full_attention"] * 5
    config.num_target_layers = 4
    config.block_size = 7
    config.dflash_config = dict(
        projector_type="dspark",
        target_layer_ids=[0, 1],
        markov_rank=8,
        late_conditioning=dict(enabled=True, bank_size=13, max_context_lag=8),
    )
    draft = draft_module.DSparkDraftModel(config)
    if legacy:
        # Legacy export has the same weight shapes and only the enabled flag.
        draft.config.dflash_config["late_conditioning"] = {"enabled": True}
    draft.save_pretrained(draft_dir)
    prompts = [{"prompt_token_ids": [1, 3, 5, i + 7]} for i in range(5)]
    params = SamplingParams(temperature=0, max_tokens=24, ignore_eos=True)
    kwargs = dict(
        skip_tokenizer_init=True,
        dtype="bfloat16",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.5,
        disable_log_stats=False,
    )
    with vllm_runner(str(target_dir), **kwargs) as runner:
        expected = [
            o.outputs[0].token_ids for o in runner.llm.generate(prompts, params)
        ]
    with vllm_runner(
        str(target_dir),
        **kwargs,
        speculative_config=dict(
            method="dspark",
            model=str(draft_dir),
            num_speculative_tokens=7,
            draft_sample_method=draft_sample_method,
        ),
    ) as runner:
        result = runner.llm.generate(prompts, params)
        assert [o.outputs[0].token_ids for o in result] == expected
        # Temperature one exercises cached proposal logits/rejection, too.
        sampled = runner.llm.generate(
            prompts,
            SamplingParams(
                temperature=1,
                top_p=1,
                top_k=-1,
                seed=42,
                max_tokens=24,
                ignore_eos=True,
            ),
        )
        assert all(len(o.outputs[0].token_ids) == 24 for o in sampled)
