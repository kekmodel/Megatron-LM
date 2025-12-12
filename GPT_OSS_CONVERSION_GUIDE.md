# YaRN RoPE Patch for Megatron-LM v0.15.0

General YaRN support for GPT-OSS, Qwen, and other models.

## Quick Start

### 1. Megatron-LM Patch (필수)

```bash
cd /path/to/Megatron-LM
git apply patches/megatron_yarn_general.patch
```

### 2. mbridge Patch (GPT-OSS 변환 시)

```bash
MBRIDGE_GPT_OSS=$(python3 -c "import mbridge.models.gpt_oss as m; print(m.__file__)")
sed -i 's/yarn_original_max_position_embeddings: int = 131072/yarn_original_max_position_embeddings: int = 4096/' "$MBRIDGE_GPT_OSS"
```

### 3. SLIME Patch (SLIME 사용 시)

```bash
# max_position_embeddings 덮어쓰기 방지
SLIME_ARGS="/path/to/slime/slime/backends/megatron_utils/arguments.py"
sed -i 's/args\.max_position_embeddings = args\.seq_length/args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length/' "$SLIME_ARGS"

# model_provider v0.15.0 호환 (**kwargs 추가)
SLIME_MP="/path/to/slime/slime/backends/megatron_utils/model_provider.py"
sed -i 's/def model_provider(pre_process=True, post_process=True, vp_stage=None):/def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):/' "$SLIME_MP"
```

---

## CLI Arguments

### GPT-OSS (프리셋)

```bash
--enable-gpt-oss \
--max-position-embeddings 131072
```

### GPT-OSS (수동)

```bash
--position-embedding-type yarn \
--yarn-original-max-position-embeddings 4096 \
--yarn-rotary-scaling-factor 32.0 \
--max-position-embeddings 131072
```

### Qwen (예시)

```bash
--position-embedding-type yarn \
--yarn-original-max-position-embeddings 32768 \
--yarn-rotary-scaling-factor 4.0 \
--max-position-embeddings 131072
```

### 모든 YaRN CLI 인자

| 인자 | 설명 | GPT-OSS 기본값 |
|------|------|---------------|
| `--yarn-rotary-scaling-factor` | YaRN 스케일링 팩터 | 32.0 |
| `--yarn-original-max-position-embeddings` | 원래 학습 길이 (확장 전) | 4096 |
| `--yarn-beta-fast` | YaRN beta_fast | 32.0 |
| `--yarn-beta-slow` | YaRN beta_slow | 1.0 |
| `--yarn-mscale` | YaRN mscale | 1.0 |
| `--yarn-mscale-all-dim` | YaRN mscale_all_dim | 0.0 |

---

## YaRN 개념

```
original_max_position_embeddings = 4096   ← 원래 학습된 컨텍스트 길이
yarn_rotary_scaling_factor = 32.0         ← 확장 비율
max_position_embeddings = 131072          ← 확장된 길이 (4096 × 32)
```

**중요**: `yarn_original_max_position_embeddings`는 확장 **전** 길이 (4096)이지, 확장 **후** 길이 (131072)가 아님!

---

## 패치 내용 요약

### megatron_yarn_general.patch

| 파일 | 변경 내용 |
|------|----------|
| `gpt_model.py` | `position_embedding_type=self.position_embedding_type` 버그 수정 |
| `gpt_model.py` | YaRN Optional 값 처리 (None → 기본값) |
| `transformer_config.py` | yarn_* 필드 Optional로 변경 |
| `arguments.py` | `--yarn-*` CLI 인자 추가 |
| `arguments.py` | `--enable-gpt-oss` 프리셋 로직 개선 |
| `model_builder.py` | 중복 코드 제거 |

### mbridge fix

```python
# Before (버그)
yarn_original_max_position_embeddings: int = 131072

# After (수정)
yarn_original_max_position_embeddings: int = 4096
```

### SLIME fix

1. `max_position_embeddings` 덮어쓰기 방지 (CLI 값 유지)
2. `model_provider` v0.15.0 호환 (`**kwargs` 추가)

---

## GPT-OSS Model Specifications

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
| `add_bias_linear` | True | True |

---

## Conversion Example (SLIME)

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
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights \
    --padded-vocab-size 201088
```

---

## Dockerfile Example

```dockerfile
# Megatron patch only (mbridge/SLIME는 수동 패치)
COPY patches/megatron_yarn_general.patch /root/Megatron-LM/megatron.patch
RUN cd Megatron-LM && \
    git apply megatron.patch --3way && \
    rm megatron.patch
```

---

## Troubleshooting

### `NotImplementedError: Unsupported parameter name: embedding.position_embeddings.weight`

**원인**: `position_embedding_type`이 `yarn`이 아닌 `learned_absolute`로 설정됨

**해결**:
1. `--enable-gpt-oss` 플래그 확인
2. Megatron 패치 적용 확인 (`position_embedding_type=self.position_embedding_type`)

### `yarn_original_max_position_embeddings = 131072`

**원인**: mbridge GPTOSSConfig 버그

**해결**: mbridge sed 패치 적용
