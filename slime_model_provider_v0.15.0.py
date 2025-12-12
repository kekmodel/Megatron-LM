"""
SLIME model_provider for Megatron-LM v0.15.0
- GPT-OSS YaRN support via --enable-gpt-oss flag
- Compatible with v0.15.0 API (accepts **kwargs)
"""

import argparse
from contextlib import nullcontext
from typing import Literal

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import core_transformer_config_from_args


def get_model_provider_func(args: argparse.Namespace):
    """Returns model_provider function for SLIME conversion."""

    def model_provider(
        pre_process: bool = True,
        post_process: bool = True,
        vp_stage: int | None = None,
        **kwargs  # v0.15.0 compatibility
    ) -> GPTModel:
        """Builds GPT model for HF to Megatron conversion."""

        use_te = args.transformer_impl == "transformer_engine"

        # Get config from args
        # NOTE: --enable-gpt-oss automatically sets YaRN config in core_transformer_config_from_args
        config: TransformerConfig = core_transformer_config_from_args(args)

        # Build transformer layer spec
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
            if callable(transformer_layer_spec):
                transformer_layer_spec = transformer_layer_spec(args, config, vp_stage)
        else:
            if args.num_experts:
                spec_kwargs = {"use_transformer_engine": use_te}
                if vp_stage is not None:
                    spec_kwargs["vp_stage"] = vp_stage
                transformer_layer_spec = get_gpt_decoder_block_spec(config, **spec_kwargs)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts,
                        args.moe_grouped_gemm,
                        args.qk_layernorm,
                        args.multi_latent_attention,
                        args.moe_use_legacy_grouped_gemm,
                    )

        # Build GPTModel kwargs
        # NOTE: position_embedding_type and rope_scaling are NOT passed here
        #       They come from config (set by --enable-gpt-oss)
        model_kwargs = {
            "config": config,
            "transformer_layer_spec": transformer_layer_spec,
            "vocab_size": args.padded_vocab_size,
            "max_sequence_length": args.max_position_embeddings,
            "pre_process": pre_process,
            "post_process": post_process,
            "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
            "parallel_output": True,
            "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
            "rotary_percent": args.rotary_percent,
            "rotary_base": args.rotary_base,
        }

        if vp_stage is not None:
            model_kwargs["vp_stage"] = vp_stage

        # Create model
        model = GPTModel(**model_kwargs)

        return model

    return model_provider
