# YaRN RoPE Patch for Megatron-LM v0.15.0

General YaRN support for GPT-OSS, Qwen, and other models.

---

## 수정 파일 상세 (File Modification Details)

### 1. Megatron-LM

| 파일 경로 | 수정 위치 | 변경 내용 |
|----------|----------|----------|
| `megatron/core/models/gpt/gpt_model.py` | Line 147 | `position_embedding_type=position_embedding_type` → `position_embedding_type=self.position_embedding_type` |
| `megatron/core/models/gpt/gpt_model.py` | Line 165-189 | YaRN config None 처리 추가 (기본값 적용) |
| `megatron/core/transformer/transformer_config.py` | Line 195-216 | `yarn_*` 필드 타입 `Optional`로 변경, 기본값 `None` |
| `megatron/training/arguments.py` | Line 2238-2251 | `--yarn-*` CLI 인자 추가 |
| `megatron/training/arguments.py` | Line 1331-1365 | `core_transformer_config_from_args`에서 YaRN 설정 처리 |
| `megatron/post_training/model_builder.py` | Line 151-164 | 중복 YaRN 설정 코드 제거 |

**적용 방법:**
```bash
cd /path/to/Megatron-LM
git apply patches/megatron_yarn_general.patch
```

---

### 2. mbridge

| 파일 경로 | 수정 위치 | 변경 내용 |
|----------|----------|----------|
| `mbridge/models/gpt_oss.py` | `GPTOSSConfig` 클래스 | `yarn_original_max_position_embeddings: int = 131072` → `4096` |

**적용 방법:**
```bash
MBRIDGE_GPT_OSS=$(python3 -c "import mbridge.models.gpt_oss as m; print(m.__file__)")
sed -i 's/yarn_original_max_position_embeddings: int = 131072/yarn_original_max_position_embeddings: int = 4096/' "$MBRIDGE_GPT_OSS"

# 확인
grep "yarn_original_max_position_embeddings" "$MBRIDGE_GPT_OSS"
```

**Before:**
```python
@dataclass
class GPTOSSConfig(TransformerConfig):
    yarn_original_max_position_embeddings: int = 131072  # ❌ 버그
```

**After:**
```python
@dataclass
class GPTOSSConfig(TransformerConfig):
    yarn_original_max_position_embeddings: int = 4096   # ✅ 수정
```

---

### 3. SLIME

| 파일 경로 | 수정 위치 | 변경 내용 |
|----------|----------|----------|
| `slime/backends/megatron_utils/arguments.py` | `set_default_megatron_args` 함수 | `max_position_embeddings` 덮어쓰기 방지 |
| `slime/backends/megatron_utils/model_provider.py` | `model_provider` 함수 시그니처 | `**kwargs` 추가 (v0.15.0 호환) |

**적용 방법:**
```bash
# 1. arguments.py - max_position_embeddings 덮어쓰기 방지
SLIME_ARGS="/path/to/slime/slime/backends/megatron_utils/arguments.py"
sed -i 's/args\.max_position_embeddings = args\.seq_length/args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length/' "$SLIME_ARGS"

# 2. model_provider.py - **kwargs 추가
SLIME_MP="/path/to/slime/slime/backends/megatron_utils/model_provider.py"
sed -i 's/def model_provider(pre_process=True, post_process=True, vp_stage=None):/def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):/' "$SLIME_MP"

# 확인
grep "max_position_embeddings" "$SLIME_ARGS"
grep "def model_provider" "$SLIME_MP"
```

**arguments.py Before:**
```python
def set_default_megatron_args(args):
    if not hasattr(args, 'seq_length') or args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length  # ❌ 항상 덮어씀
```

**arguments.py After:**
```python
def set_default_megatron_args(args):
    if not hasattr(args, 'seq_length') or args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length  # ✅ CLI 값 유지
```

**model_provider.py Before:**
```python
def model_provider(pre_process=True, post_process=True, vp_stage=None):  # ❌ v0.15.0 비호환
```

**model_provider.py After:**
```python
def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):  # ✅ v0.15.0 호환
```

---

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
SLIME_ARGS="/path/to/slime/slime/backends/megatron_utils/arguments.py"
sed -i 's/args\.max_position_embeddings = args\.seq_length/args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length/' "$SLIME_ARGS"

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
# Megatron patch
COPY patches/megatron_yarn_general.patch /root/Megatron-LM/megatron.patch
RUN cd Megatron-LM && git apply megatron.patch --3way && rm megatron.patch

# mbridge patch
RUN MBRIDGE_GPT_OSS=$(python3 -c "import mbridge.models.gpt_oss as m; print(m.__file__)") && \
    sed -i 's/yarn_original_max_position_embeddings: int = 131072/yarn_original_max_position_embeddings: int = 4096/' "$MBRIDGE_GPT_OSS"

# SLIME patch
RUN sed -i 's/args\.max_position_embeddings = args\.seq_length/args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length/' \
    /path/to/slime/slime/backends/megatron_utils/arguments.py
RUN sed -i 's/def model_provider(pre_process=True, post_process=True, vp_stage=None):/def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):/' \
    /path/to/slime/slime/backends/megatron_utils/model_provider.py
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

### `max_position_embeddings`가 `seq_length`로 덮어써짐

**원인**: SLIME `set_default_megatron_args`에서 강제 덮어쓰기

**해결**: SLIME arguments.py sed 패치 적용

### v0.15.0에서 `model_provider() got unexpected keyword argument 'config'`

**원인**: Megatron v0.15.0이 `model_provider`에 `config=`, `pg_collection=` 전달

**해결**: SLIME model_provider.py에 `**kwargs` 추가
