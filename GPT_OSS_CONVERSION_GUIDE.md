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

> **Important**: The `--enable-gpt-oss` flag had a bug setting `yarn_original_max_position_embeddings=131072` instead of `4096`. This patch fixes it.

## Conversion Commands

### Single GPU Conversion (GPT-OSS-20B)

```bash
python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint openai/gpt-oss-20b \
    --save /path/to/output \
    --bf16 \
    --use-mcore-models \
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
    --enable-gpt-oss \
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
    --enable-gpt-oss \
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

### Position Embedding
| Argument | Value | Description |
|----------|-------|-------------|
| `--enable-gpt-oss` | - | Auto-configure YaRN (patch required!) |
| `--max-position-embeddings` | 131072 | Extended context length |
| `--rotary-base` | 150000 | RoPE base frequency |
| `--rotary-percent` | 1.0 | Full rotary embedding |

### MoE Configuration
| Argument | Value | Description |
|----------|-------|-------------|
| `--num-experts` | 32/128 | Number of experts |
| `--moe-router-topk` | 4 | Experts per token |
| `--moe-router-dtype` | fp32 | Router precision |

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

## Bug Fix Details

### Megatron `--enable-gpt-oss` Bug (Fixed by patch)

The patch fixes `megatron/post_training/model_builder.py`:

```python
# Before (incorrect)
config.yarn_original_max_position_embeddings = 131072

# After (correct)
config.yarn_original_max_position_embeddings = 4096
```

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

### Checkpoint format issues
- The converted checkpoint is saved in "release" format
- Compatible with Megatron distributed checkpointing

## References

- [OpenAI GPT-OSS Repository](https://github.com/openai/gpt-oss)
- [HuggingFace GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [HuggingFace GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)
- [ISEEKYAN mbridge](https://github.com/ISEEKYAN/mbridge)
- [NVIDIA Megatron-Bridge Docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/llm/gpt-oss.html)
