# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for draft config overrides used by SpeculativeConfig.

Callable ``hf_overrides`` on the target model config (e.g. the
``dummy_hf_overrides`` shrink used by ``tests/models/test_initialization.py``)
must also be applied when building the draft ``ModelConfig``. Otherwise a
draft belonging to a large target model is instantiated at full size even
when the target itself is shrunk — which is what kept spec-decode archs like
``EagleMistralLarge3ForCausalLM`` stuck at ``is_available_online=False``
("TODO: revert once figuring out OOM in CI").
"""

import functools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers import PreTrainedConfig

from vllm.config.parallel import ParallelConfig
from vllm.config.speculative import SpeculativeConfig, _validate_dspark_ngram_config


def _make_hf_config(**kwargs) -> PreTrainedConfig:
    defaults = dict(
        architectures=["LlamaForCausalLM"],
        model_type="llama",
        num_hidden_layers=64,
    )
    defaults.update(kwargs)
    return PreTrainedConfig(**defaults)


@pytest.mark.cpu_test
def test_dict_overrides_are_not_forwarded_to_draft():
    """Dict overrides are target-specific key patches; the draft must get
    only the architecture-mapping override."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(
        {"max_position_embeddings": 1234}
    )
    assert composed is SpeculativeConfig.hf_config_override


@pytest.mark.cpu_test
def test_none_overrides_fall_back_to_arch_mapping():
    composed = SpeculativeConfig.compose_draft_hf_overrides(None)
    assert composed is SpeculativeConfig.hf_config_override


def _ngram_config():
    return SimpleNamespace(
        method="dspark",
        draft_model_config=SimpleNamespace(
            architectures=["Qwen3DSparkModel"],
            hf_config=SimpleNamespace(
                dflash_config={"markov_head_type": "ngram_attention"}
            ),
        ),
        target_model_config=SimpleNamespace(
            enforce_eager=True, hf_text_config=SimpleNamespace(ple_layer_ids=[2])
        ),
        target_parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            tensor_parallel_size=4,
        ),
        draft_tensor_parallel_size=4,
        draft_sample_method="greedy",
        enable_adaptive_verification=False,
        dspark_draft_topk=None,
    )


def test_ngram_markov_accepts_eager_target_table_reuse():
    _validate_dspark_ngram_config(_ngram_config())


def test_ngram_validation_does_not_change_vanilla_dspark():
    config = _ngram_config()
    config.draft_model_config.hf_config.dflash_config = {"markov_head_type": "vanilla"}
    config.target_model_config.enforce_eager = False
    config.draft_tensor_parallel_size = 1
    _validate_dspark_ngram_config(config)
    _validate_dspark_ngram_config(SimpleNamespace(method="mtp"))


@pytest.mark.parametrize(
    "setting, value, message",
    [
        ("prefix_reranker", {"top_k": 16}, "prefix reranking"),
        ("sample_from_anchor", False, "sample_from_anchor"),
    ],
)
def test_ngram_markov_rejects_incompatible_draft_structure(setting, value, message):
    config = _ngram_config()
    setattr(config.draft_model_config.hf_config, setting, value)
    with pytest.raises(ValueError, match=message):
        _validate_dspark_ngram_config(config)


@pytest.mark.parametrize(
    "section, name, value, message",
    [
        (None, "draft_sample_method", "probabilistic", "greedy drafting"),
        (None, "enable_adaptive_verification", True, "Disable adaptive"),
        (None, "dspark_draft_topk", 16, "Do not combine"),
        (None, "draft_tensor_parallel_size", 1, "draft TP=target TP"),
        ("target_model_config", "enforce_eager", False, "enforce-eager"),
        ("target_parallel_config", "pipeline_parallel_size", 2, "PP=DP"),
        ("target_parallel_config", "data_parallel_size", 2, "PP=DP"),
        ("target_parallel_config", "prefill_context_parallel_size", 2, "PP=DP"),
        ("target_parallel_config", "decode_context_parallel_size", 2, "PP=DP"),
    ],
)
def test_ngram_markov_rejects_unsupported_execution(section, name, value, message):
    config = _ngram_config()
    setattr(getattr(config, section) if section else config, name, value)
    with pytest.raises(ValueError, match=message):
        _validate_dspark_ngram_config(config)


@pytest.mark.parametrize("ple_layers", [[], [1], [2, 4]])
def test_ngram_markov_requires_the_cache_source_ple_layer(ple_layers):
    config = _ngram_config()
    config.target_model_config.hf_text_config.ple_layer_ids = ple_layers
    with pytest.raises(ValueError, match="ple_layer_ids"):
        _validate_dspark_ngram_config(config)


def _make_speculative_config(
    hf_kwargs: dict, architecture: str, **speculative_kwargs
) -> SpeculativeConfig:
    hf_config = SpeculativeConfig.hf_config_override(_make_hf_config(**hf_kwargs))
    draft = MagicMock(
        hf_config=hf_config, architectures=hf_config.architectures, max_model_len=128
    )
    draft.registry.inspect_model_cls.return_value = (None, architecture)
    target = MagicMock(max_model_len=128, quantization=None, hf_overrides={})
    target.hf_config.model_type = hf_kwargs["model_type"]

    with patch("vllm.config.speculative.ModelConfig", return_value=draft):
        return SpeculativeConfig(
            model="draft",
            **speculative_kwargs,
            target_model_config=target,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("hf_kwargs", "n_predict", "architecture"),
    [
        (
            dict(
                model_type="deepseek_v4",
                num_nextn_predict_layers=3,
                dspark_block_size=5,
            ),
            3,
            "DSparkDraftModel",
        ),
        (
            dict(
                model_type="deepseek_v41",
                num_nextn_predict_layers=3,
                dspark_block_size=5,
            ),
            3,
            "DSparkV41DraftModel",
        ),
        (
            dict(
                model_type="gemma4_text",
                architectures=["Gemma4DSparkModel"],
                block_size=5,
            ),
            None,
            "Gemma4DSparkModel",
        ),
    ],
)
@pytest.mark.parametrize("speculative_kwargs", [{"num_speculative_tokens": 5}, {}])
def test_dspark_width_is_independent_of_mtp_stages(
    hf_kwargs, n_predict, architecture, speculative_kwargs
):
    config = _make_speculative_config(
        hf_kwargs, architecture, method="dspark", **speculative_kwargs
    )

    assert config.num_speculative_tokens == 5
    assert getattr(config.draft_model_config.hf_config, "n_predict", None) == n_predict
    assert config.draft_model_config.hf_config.architectures == [architecture]


@pytest.mark.cpu_test
def test_mtp_stages_are_independent_of_dspark_width():
    hf_kwargs = dict(
        model_type="deepseek_v4", num_nextn_predict_layers=3, dspark_block_size=5
    )
    config = _make_speculative_config(
        hf_kwargs, "DeepSeekV4MTPModel", method="mtp", num_speculative_tokens=6
    )

    assert config.num_speculative_tokens == 6
    assert config.draft_model_config.hf_config.n_predict == 3
    assert config.draft_model_config.hf_config.architectures == ["DeepSeekV4MTPModel"]

    with pytest.raises(ValueError, match="must be divisible by n_predict=3"):
        _make_speculative_config(
            hf_kwargs, "DeepSeekV4MTPModel", method="mtp", num_speculative_tokens=5
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"draft_sample_method": "probabilistic"}, "greedy drafting only"),
        ({"dspark_draft_topk": 4}, "Do not combine"),
        ({"enable_adaptive_verification": True}, "Disable adaptive verification"),
        ({"num_speculative_tokens": 5}, "exceeds reranker block_size"),
    ],
)
def test_prefix_reranker_rejects_incompatible_sampling_modes(overrides, message):
    hf_kwargs = dict(
        model_type="qwen3",
        architectures=["Qwen3DSparkModel"],
        vocab_size=32,
        block_size=4,
        dflash_config={"prefix_reranker": {"width": 8, "num_heads": 2, "top_k": 3}},
    )
    settings = {"method": "dspark", "num_speculative_tokens": 4, **overrides}
    with pytest.raises(ValueError, match=message):
        _make_speculative_config(hf_kwargs, "Qwen3DSparkModel", **settings)


@pytest.mark.cpu_test
def test_callable_overrides_reach_the_draft_config():
    """A callable override (config-to-config transform) composes with the
    architecture-mapping override and is applied to the draft config."""

    def shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        hf_config.num_hidden_layers = 1
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(shrink)
    assert composed is not SpeculativeConfig.hf_config_override

    out = composed(_make_hf_config())
    # The shrink transform must have been applied to the draft config.
    assert out.num_hidden_layers == 1


@pytest.mark.cpu_test
def test_arch_mapping_applies_before_callable_override():
    """The static arch-mapping override runs first, so the user callable
    observes (and may adjust) the post-mapping config."""
    seen_architectures: list[str] = []

    def record(hf_config: PreTrainedConfig) -> PreTrainedConfig:
        seen_architectures.append(hf_config.architectures[0])
        return hf_config

    composed = SpeculativeConfig.compose_draft_hf_overrides(record)

    # MiMo is one of the arch-mapped model types: hf_config_override
    # rewrites architectures to ["MiMoMTPModel"].
    mimo = _make_hf_config(
        architectures=["MiMoForCausalLM"],
        model_type="mimo",
        num_nextn_predict_layers=1,
    )
    composed(mimo)
    assert seen_architectures == ["MiMoMTPModel"]


@pytest.mark.cpu_test
def test_inkling_override_exposes_all_mtp_depths():
    text_config = _make_hf_config(
        architectures=["InklingForCausalLM"],
        model_type="inkling_model",
        local_layer_ids=[1, 3],
    )
    config = _make_hf_config(
        architectures=["InklingForConditionalGeneration"],
        model_type="inkling_mm_model",
        text_config=text_config,
        mtp_config={
            "num_nextn_predict_layers": 8,
            "local_layer_ids": [0, 2, 4],
        },
    )

    out = SpeculativeConfig.hf_config_override(config)

    assert out is text_config
    assert out.model_type == "inkling_mtp"
    assert out.architectures == ["InklingMTPModel"]
    # Multi-module MTP: every checkpoint depth is exposed (module i drafts
    # speculative token i), no longer clamped to the first depth.
    assert out.n_predict == 8
    assert out.num_nextn_predict_layers == 8
    assert out.chain_hidden_post_norm is False
    assert out.local_layer_ids == [0, 2, 4]


def _module_level_shrink(hf_config: PreTrainedConfig) -> PreTrainedConfig:
    hf_config.num_hidden_layers = 1
    return hf_config


@pytest.mark.cpu_test
def test_composed_override_is_picklable():
    """The draft ``ModelConfig`` is sent to spawned engine-core processes, so
    the composed override must be picklable. A nested local closure is not
    (it raised ``Can't get local object`` on DFlashDraftModel); a
    ``functools.partial`` over a module-referenceable static method is.
    Guard against regressing to a closure."""
    composed = SpeculativeConfig.compose_draft_hf_overrides(_module_level_shrink)

    assert isinstance(composed, functools.partial)
    assert composed.func is SpeculativeConfig._apply_composed_hf_override

    out = composed(_make_hf_config())
    assert out.num_hidden_layers == 1


def _make_mtp_speculative_config(
    override: bool | None,
    checkpoint_value: bool,
) -> SpeculativeConfig:
    draft_hf_config = _make_hf_config(
        architectures=["Qwen4ExpMTP"],
        model_type="qwen4_exp_mtp",
        n_predict=1,
        index_share_for_mtp_iteration=checkpoint_value,
    )
    draft_model_config = MagicMock(
        model="draft",
        hf_config=draft_hf_config,
        architectures=draft_hf_config.architectures,
        max_model_len=128,
    )
    target_model_config = MagicMock(
        model="target",
        max_model_len=128,
        quantization=None,
        hf_overrides={},
    )

    with patch("vllm.config.speculative.ModelConfig", return_value=draft_model_config):
        return SpeculativeConfig(
            model="draft",
            method="mtp",
            num_speculative_tokens=1,
            index_share_for_mtp_iteration=override,
            target_model_config=target_model_config,
            target_parallel_config=ParallelConfig(),
        )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("override", "checkpoint_value", "expected"),
    [(None, True, True), (False, True, False), (True, False, True)],
)
def test_mtp_index_share_override(
    override: bool | None, checkpoint_value: bool, expected: bool
):
    speculative_config = _make_mtp_speculative_config(override, checkpoint_value)
    assert (
        speculative_config.draft_model_config.hf_config.index_share_for_mtp_iteration
        is expected
    )
