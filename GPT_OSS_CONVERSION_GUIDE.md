# GPT-OSS Conversion Guide for Megatron-LM v0.15.0

This guide explains how to convert OpenAI GPT-OSS models from HuggingFace format to Megatron-LM checkpoint format.

## Prerequisites

1. **Apply the v0.15.0 patch** (includes GPT-OSS YaRN bug fix):
```bash
cd /path/to/Megatron-LM
git apply megatron_v0.15.0.patch
```

2. **Install mbridge with GPT-OSS support**:
```bash
pip install git+https://github.com/ISEEKYAN/mbridge.git@gpt-oss --no-deps
```

> ⚠️ **Known Issue**: mbridge also has incorrect `yarn_original_max_position_embeddings=131072` default. The patch fixes Megatron side, but mbridge may need separate fix for full compatibility.

## Model Specifications

| Parameter | GPT-OSS-20B | GPT-OSS-120B |
|-----------|-------------|--------------|
| `num_layers` | 24 | 36 |
| `hidden_size` | 2880 | 2880 |
| `ffn_hidden_size` | 2880 | 2880 |
| `num_attention_heads` | 64 | 64 |
| `num_query_groups` | 8 | 8 |
| `kv_channels` | 64 | 64 |
| `num_experts` | 32 | 128 |
| `moe_router_topk` | 4 | 4 |
| `max_position_embeddings` | 131072 | 131072 |
| `vocab_size` | 201088 | 201088 |

## YaRN RoPE Configuration

GPT-OSS uses YaRN (Yet another RoPE extensioN) to extend context length:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `original_max_position_embeddings` | **4096** | Original training context (NOT 131072!) |
| `factor` | 32.0 | Scaling factor |
| `max_position_embeddings` | 131072 | Extended context (4096 × 32) |
| `beta_fast` | 32.0 | YaRN frequency parameter |
| `beta_slow` | 1.0 | YaRN frequency parameter |
| `rope_base` | 150000 | RoPE base frequency |

> ✅ **v0.15.0 패치 적용 후**: `--enable-gpt-oss` 플래그 사용 가능! TransformerConfig에 YaRN 기본값이 포함됨.

## Conversion Commands (SLIME)

> **Note**: `--enable-gpt-oss` 플래그가 자동으로 YaRN 설정을 적용함 (v0.15.0 패치 필요)

### Single GPU Conversion (GPT-OSS-20B)

```bash
python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint openai/gpt-oss-20b \
    --save /path/to/output \
    --bf16 \
    --use-mcore-models \
    --enable-gpt-oss \
    --num-layers 24 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 64 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --rotary-base 150000 \
    --rotary-percent 1.0 \
    --num-experts 32 \
    --moe-ffn-hidden-size 2880 \
    --moe-router-topk 4 \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --untie-embeddings-and-output-weights \
    --padded-vocab-size 201088 \
    --quick-geglu \
    --glu-linear-offset 1.0 \
    --activation-func-clamp-value 7.0 \
    --softmax-type learnable \
    --window-size 128,0 \
    --window-attn-skip-freq 2 \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion
```

### Multi-GPU Conversion (Tensor Parallel)

```bash
torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint openai/gpt-oss-20b \
    --save /path/to/output \
    --tensor-model-parallel-size 8 \
    --bf16 \
    --use-mcore-models \
    --enable-gpt-oss \
    --num-layers 24 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 64 \
    --seq-length 4096 \
    --max-position-embeddings 131072 \
    --rotary-base 150000 \
    --rotary-percent 1.0 \
    --num-experts 32 \
    --moe-ffn-hidden-size 2880 \
    --moe-router-topk 4 \
    --moe-router-dtype fp32 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --untie-embeddings-and-output-weights \
    --padded-vocab-size 201088 \
    --quick-geglu \
    --glu-linear-offset 1.0 \
    --activation-func-clamp-value 7.0 \
    --softmax-type learnable \
    --window-size 128,0 \
    --window-attn-skip-freq 2 \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion
```

### GPT-OSS-120B Conversion

For GPT-OSS-120B, modify the following parameters:

```bash
    --hf-checkpoint openai/gpt-oss-120b \
    --num-layers 36 \
    --num-experts 128 \
    # ... rest same as above
```

## Key Arguments Explained

### Architecture Parameters
| Argument | Value | Description |
|----------|-------|-------------|
| `--num-layers` | 24/36 | Number of transformer layers |
| `--hidden-size` | 2880 | Hidden dimension |
| `--kv-channels` | 64 | Key/Value head dimension |
| `--num-query-groups` | 8 | GQA groups (num_key_value_heads) |

### Position Embedding (CLI)
| Argument | Value | Description |
|----------|-------|-------------|
| `--position-embedding-type` | rope | Use RoPE (code overrides to YaRN) |
| `--max-position-embeddings` | 131072 | Extended context length |
| `--rotary-base` | 150000 | RoPE base frequency |
| `--rotary-percent` | 1.0 | Full rotary embedding |

> ⚠️ **Note**: `--use-rope-scaling` and `--rope-scaling-factor` are **Llama3.x style**, not YaRN. Don't use them for GPT-OSS. Set YaRN config in code instead.

### MoE Configuration
| Argument | Value | Description |
|----------|-------|-------------|
| `--num-experts` | 32/128 | Number of experts |
| `--moe-router-topk` | 4 | Experts per token |
| `--moe-router-dtype` | fp32 | Router precision |
| `--moe-grouped-gemm` | - | **Required for mbridge** (uses GroupedMLP) |

### Activation & Normalization
| Argument | Value | Description |
|----------|-------|-------------|
| `--quick-geglu` | - | GPT-OSS activation |
| `--glu-linear-offset` | 1.0 | GLU offset |
| `--activation-func-clamp-value` | 7.0 | swiglu_limit |
| `--norm-epsilon` | 1e-5 | RMSNorm epsilon |
| `--softmax-type` | learnable | Learnable softmax offset |

### Attention
| Argument | Value | Description |
|----------|-------|-------------|
| `--window-size` | 128,0 | Sliding window |
| `--window-attn-skip-freq` | 2 | Full attention frequency |

### Fusion Flags (Recommended)
| Argument | Description |
|----------|-------------|
| `--no-masked-softmax-fusion` | Disable masked softmax fusion |
| `--no-rope-fusion` | Disable RoPE fusion |
| `--no-bias-gelu-fusion` | Disable bias-gelu fusion |
| `--no-bias-dropout-fusion` | Disable bias-dropout fusion |

## YaRN Configuration for SLIME Users

SLIME uses its own `convert_hf_to_torch_dist.py` which doesn't have `--enable-gpt-oss`.
YaRN parameters are **not configurable via CLI** (only MLA supports `--rope-type yarn`).

### CLI vs Code Configuration

| Parameter | CLI Support? | Notes |
|-----------|-------------|-------|
| `--position-embedding-type rope` | ✅ | Use `rope` in CLI |
| `--rotary-base 150000` | ✅ | |
| `--rotary-percent 1.0` | ✅ | |
| `--max-position-embeddings 131072` | ✅ | |
| `--use-rope-scaling` | ❌ | **Don't use** - Llama3.x style, not YaRN |
| `config.position_embedding_type = "yarn"` | ❌ | **Code only** (overwrites CLI) |
| `config.yarn_original_max_position_embeddings` | ❌ | **Code only** - 핵심! |
| `config.yarn_beta_fast`, `yarn_beta_slow` | ❌ | **Code only** |
| `config.yarn_mscale`, `yarn_mscale_all_dim` | ❌ | **Code only** |
| `config.yarn_correction_range_round_to_int` | ❌ | **Code only** - v0.15.0 필수! |

### Required Code Changes in model_provider

**Step 1**: Set position_embedding_type to "yarn" **after** `core_transformer_config_from_args(args)`:

```python
config: TransformerConfig = core_transformer_config_from_args(args)

# GPT-OSS YaRN - TransformerConfig에 기본값이 있으므로 이것만 설정!
config.position_embedding_type = "yarn"  # CLI는 'rope'만 지원, yarn으로 덮어쓰기

# 아래는 TransformerConfig 기본값 (필요시 오버라이드):
# config.yarn_rotary_scaling_factor = 32.0  (기본값)
# config.yarn_original_max_position_embeddings = 4096  (기본값)
# config.yarn_beta_fast = 32.0  (기본값)
# config.yarn_beta_slow = 1.0  (기본값)
# config.yarn_mscale = 1.0  (기본값)
# config.yarn_mscale_all_dim = 0.0  (기본값)
# config.yarn_correction_range_round_to_int = False  (기본값)
```

> ✅ **v0.15.0 패치 적용 후**: TransformerConfig에 GPT-OSS YaRN 기본값이 추가됨. `position_embedding_type = "yarn"`만 설정하면 됨!

**Step 2**: Remove conflicting kwargs from GPTModel (kwargs가 config보다 우선순위 높음!):

```python
kwargs = {
    "config": config,
    "transformer_layer_spec": transformer_layer_spec,
    "vocab_size": args.padded_vocab_size,
    "max_sequence_length": args.max_position_embeddings,
    "pre_process": pre_process,
    "post_process": post_process,
    "fp16_lm_cross_entropy": args.fp16_lm_cross_entropy,
    "parallel_output": True,
    "share_embeddings_and_output_weights": not args.untie_embeddings_and_output_weights,
    # ❌ 삭제: "position_embedding_type": args.position_embedding_type,
    "rotary_percent": args.rotary_percent,
    "rotary_base": args.rotary_base,
    # ❌ 삭제: "rope_scaling": args.use_rope_scaling,
}
```

> ⚠️ **Why remove from kwargs?** GPTModel uses kwargs over config:
> ```python
> self.position_embedding_type = (
>     position_embedding_type if position_embedding_type is not None
>     else self.config.position_embedding_type
> )
> ```

## Bug Fix Details

### Megatron `--enable-gpt-oss` Bug

The `--enable-gpt-oss` flag in `megatron/post_training/model_builder.py` **had** a bug:

```python
# Bug (before patch): sets wrong value
config.yarn_original_max_position_embeddings = 131072  # Wrong! Should be 4096
```

**After applying `megatron_v0.15.0.patch`**: The bug is fixed. But `--enable-gpt-oss` is only available in Megatron's `post_training` scripts, not in SLIME.

**For SLIME users**: Use manual YaRN configuration in code (see above section).

### mbridge Bug (Not fixed)

mbridge `gpt_oss.py` has similar issue:
```python
# mbridge default (incorrect)
yarn_original_max_position_embeddings: int = 131072  # Should be 4096
```

This may cause issues when mbridge reads the model. Consider patching mbridge or using the conversion script directly.

## Parameter Comparison

| Parameter | HuggingFace | Megatron Script | Correct |
|-----------|-------------|-----------------|---------|
| `max_position_embeddings` | 131072 | 40960 | **131072** |
| `rope_scaling.original_max_position_embeddings` | 4096 | - | **4096** |
| `seq_length` | - | 4096 | 4096 (or custom) |

### Why 40960?

The Megatron GPT-OSS script was copy-pasted from **Qwen3 template** without updating `max-position-embeddings`:

```bash
# Qwen3-0.6B.sh (original)
--max-position-embeddings 40960   # Qwen3's actual value

# gpt-oss-20b.sh (copy-pasted, forgot to change!)
--max-position-embeddings 40960   # Should be 131072 for GPT-OSS
```

Qwen3 natively uses 40960, but GPT-OSS uses YaRN to extend 4096 → 131072 (×32).

## Troubleshooting

### "World size must be less than or equal to number of layers"
- Use fewer GPUs or enable pipeline parallelism appropriately

### Memory issues during conversion
- The conversion uses `memory_efficient=True` by default
- For large models, consider using more GPUs with tensor parallelism

### YaRN position mismatch errors
- Ensure patch is applied with correct `yarn_original_max_position_embeddings=4096`
- Check mbridge version has correct defaults

### "Unsupported parameter name: local_experts"
- mbridge doesn't support SequentialMLP weight naming
- **Solution**: Add `--moe-grouped-gemm` flag to use GroupedMLP instead

### Checkpoint format issues
- The converted checkpoint is saved in "release" format
- Compatible with Megatron distributed checkpointing

## References

- [OpenAI GPT-OSS Repository](https://github.com/openai/gpt-oss)
- [HuggingFace GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [HuggingFace GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)
- [ISEEKYAN mbridge](https://github.com/ISEEKYAN/mbridge)
- [NVIDIA Megatron-Bridge Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/gpt-oss.html)
