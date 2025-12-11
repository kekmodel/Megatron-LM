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

## Model Specifications

| Parameter | GPT-OSS-20B | GPT-OSS-120B |
|-----------|-------------|--------------|
| `num_layers` | 24 | 36 |
| `hidden_size` | 2880 | 2880 |
| `num_attention_heads` | 64 | 64 |
| `num_query_groups` | 8 | 8 |
| `num_experts` | 32 | 128 |
| `moe_router_topk` | 4 | 4 |
| `max_position_embeddings` | 131072 | 131072 |
| `vocab_size` | 201088 | 201088 |

## YaRN RoPE Configuration

GPT-OSS uses YaRN (Yet another RoPE extensioN) to extend context length:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `original_max_position_embeddings` | 4096 | Original training context |
| `factor` | 32.0 | Scaling factor |
| `max_position_embeddings` | 131072 | Extended context (4096 × 32) |
| `beta_fast` | 32.0 | YaRN frequency parameter |
| `beta_slow` | 1.0 | YaRN frequency parameter |
| `rope_base` | 150000 | RoPE base frequency |

> **Note**: The `--enable-gpt-oss` flag automatically configures these parameters after applying the patch.

## Conversion Commands

### Single GPU Conversion

```bash
python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint openai/gpt-oss-20b \
    --save /path/to/output \
    --num-layers 24 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --num-experts 32 \
    --moe-ffn-hidden-size 2880 \
    --moe-router-topk 4 \
    --moe-token-dispatcher-type alltoall \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --padded-vocab-size 201088 \
    --max-position-embeddings 131072 \
    --rotary-base 150000 \
    --quick-geglu \
    --softmax-type learnable \
    --window-size 128,0 \
    --window-attn-skip-freq 2 \
    --enable-gpt-oss \
    --use-mcore-models \
    --bf16
```

### Multi-GPU Conversion (Tensor Parallel)

```bash
torchrun --nproc_per_node=8 tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint openai/gpt-oss-20b \
    --save /path/to/output \
    --tensor-model-parallel-size 8 \
    --num-layers 24 \
    --hidden-size 2880 \
    --ffn-hidden-size 2880 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --num-experts 32 \
    --moe-ffn-hidden-size 2880 \
    --moe-router-topk 4 \
    --moe-token-dispatcher-type alltoall \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --padded-vocab-size 201088 \
    --max-position-embeddings 131072 \
    --rotary-base 150000 \
    --quick-geglu \
    --softmax-type learnable \
    --window-size 128,0 \
    --window-attn-skip-freq 2 \
    --enable-gpt-oss \
    --use-mcore-models \
    --bf16
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

| Argument | Description |
|----------|-------------|
| `--enable-gpt-oss` | Enables GPT-OSS mode with correct YaRN configuration |
| `--quick-geglu` | Uses QuickGELU activation (GPT-OSS specific) |
| `--softmax-type learnable` | Learnable softmax offset |
| `--window-size 128,0` | Sliding window attention size |
| `--window-attn-skip-freq 2` | Full attention every 2nd layer |
| `--moe-token-dispatcher-type alltoall` | MoE communication pattern |

## Bug Fix Details

The patch fixes a bug in `megatron/post_training/model_builder.py` where `--enable-gpt-oss` incorrectly set:

```python
# Before (incorrect)
config.yarn_original_max_position_embeddings = 131072

# After (correct)
config.yarn_original_max_position_embeddings = 4096
```

This value should be the **original** context length (4096), not the **extended** context length (131072).

## Troubleshooting

### "World size must be less than or equal to number of layers"
- Use fewer GPUs or enable pipeline parallelism appropriately

### Memory issues during conversion
- The conversion uses `memory_efficient=True` by default
- For large models, consider using more GPUs with tensor parallelism

### Checkpoint format issues
- The converted checkpoint is saved in "release" format
- Compatible with Megatron distributed checkpointing

## References

- [OpenAI GPT-OSS Repository](https://github.com/openai/gpt-oss)
- [HuggingFace GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [HuggingFace GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)
- [ISEEKYAN mbridge](https://github.com/ISEEKYAN/mbridge)
