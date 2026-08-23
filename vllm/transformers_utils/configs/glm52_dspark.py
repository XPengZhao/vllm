# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from transformers import DeepseekV2Config


class Glm52DSparkConfig(DeepseekV2Config):
    """Standalone GLM-5.2 dense MLA DSpark draft config."""

    model_type = "glm52_dspark"
    has_no_defaults_at_init = True

    def __init__(self, **kwargs) -> None:
        kwargs["n_routed_experts"] = 0
        kwargs["n_shared_experts"] = 0
        kwargs["num_experts_per_tok"] = 0
        kwargs["n_routed_experts"] = 0
        rope_parameters = kwargs.get("rope_parameters")
        if rope_parameters is None:
            kwargs["rope_parameters"] = {
                "rope_type": "default",
                "rope_theta": kwargs.get("rope_theta", 8_000_000),
            }
        super().__init__(**kwargs)

        method = dict(getattr(self, "dflash_config", None) or {})
        aux_ids = list(
            getattr(self, "eagle_aux_hidden_state_layer_ids", None)
            or getattr(self, "target_layer_ids", None)
            or method.get("target_layer_ids")
            or ()
        )
        if aux_ids:
            self.eagle_aux_hidden_state_layer_ids = aux_ids
            self.target_layer_ids = aux_ids
            self.num_target_layers = len(aux_ids)
        self.target_hidden_size = int(
            getattr(self, "target_hidden_size", None) or self.hidden_size
        )
        if method.get("num_layers") is not None:
            self.num_hidden_layers = int(method["num_layers"])
        self.draft_vocab_size = int(
            getattr(self, "draft_vocab_size", None) or self.vocab_size
        )
        if method.get("markov_rank") is not None:
            self.markov_rank = int(method["markov_rank"])
        if not getattr(self, "markov_rank", 0):
            raise ValueError("GLM-5.2 DSpark requires a positive markov_rank")
        if not getattr(self, "target_layer_ids", None):
            raise ValueError("GLM-5.2 DSpark requires target_layer_ids")
