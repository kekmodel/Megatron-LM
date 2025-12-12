# YaRN RoPE Patch for Megatron-LM v0.15.0

General YaRN support for GPT-OSS, Qwen, and other models.

---

## Quick Start

```bash
# 1. Megatron-LM (필수)
cd /path/to/Megatron-LM
git apply patches/megatron_yarn_general.patch

# 2. mbridge (GPT-OSS 변환 시)
MBRIDGE_GPT_OSS=$(python3 -c "import mbridge.models.gpt_oss as m; print(m.__file__)")
sed -i 's/yarn_original_max_position_embeddings: int = 131072/yarn_original_max_position_embeddings: int = 4096/' "$MBRIDGE_GPT_OSS"

# 3. SLIME (SLIME 사용 시)
SLIME_ARGS="/path/to/slime/slime/backends/megatron_utils/arguments.py"
sed -i 's/args\.max_position_embeddings = args\.seq_length/args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length/' "$SLIME_ARGS"

SLIME_MP="/path/to/slime/slime/backends/megatron_utils/model_provider.py"
sed -i 's/def model_provider(pre_process=True, post_process=True, vp_stage=None):/def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):/' "$SLIME_MP"
```

---

## Megatron-LM 패치 상세

### 1. `megatron/core/models/gpt/gpt_model.py`

#### 변경 1: Line 147 - position_embedding_type 버그 수정 (Critical)

**문제**: kwargs로 전달된 `position_embedding_type`이 `self.position_embedding_type`을 덮어씀

**Before (Line 143-150):**
```python
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,  # ❌ kwargs 사용
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )
```

**After (Line 143-150):**
```python
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=self.position_embedding_type,  # ✅ self 사용
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
            )
```

**검증 방법:**
```bash
grep -n "position_embedding_type=self.position_embedding_type" megatron/core/models/gpt/gpt_model.py
# 출력: 147:                position_embedding_type=self.position_embedding_type,
```

---

#### 변경 2: Line 165-180 - YaRN 설정 None 처리

**문제**: `config.yarn_*` 값이 None일 때 에러 발생

**Before (Line 165-185):**
```python
        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=getattr(self.config, "yarn_rotary_scaling_factor"),
                original_max_position_embeddings=getattr(
                    self.config, "yarn_original_max_position_embeddings"
                ),
                beta_fast=getattr(self.config, "yarn_beta_fast"),
                beta_slow=getattr(self.config, "yarn_beta_slow"),
                mscale=getattr(self.config, "yarn_mscale"),
                mscale_all_dim=getattr(self.config, "yarn_mscale_all_dim"),
                correction_range_round_to_int=getattr(
                    self.config, "yarn_correction_range_round_to_int"
                ),
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
```

**After (Line 165-180):**
```python
        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=self.config.yarn_rotary_scaling_factor or 1.0,
                original_max_position_embeddings=self.config.yarn_original_max_position_embeddings or 4096,
                beta_fast=self.config.yarn_beta_fast or 32.0,
                beta_slow=self.config.yarn_beta_slow or 1.0,
                mscale=self.config.yarn_mscale or 1.0,
                mscale_all_dim=self.config.yarn_mscale_all_dim or 0.0,
                correction_range_round_to_int=self.config.yarn_correction_range_round_to_int or False,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )
```

**검증 방법:**
```bash
grep -A 15 "elif self.position_embedding_type == 'yarn':" megatron/core/models/gpt/gpt_model.py
# scaling_factor=self.config.yarn_rotary_scaling_factor or 1.0, 확인
```

---

### 2. `megatron/core/transformer/transformer_config.py`

#### 변경: Line 195-217 - YaRN 필드 추가

**문제**: YaRN 관련 config 필드가 없음

**Before (Line 192-194):**
```python
    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    test_mode: bool = False
```

**After (Line 192-219):**
```python
    qk_layernorm: bool = False
    """Whether to apply `normalization` type of normalization to the query and key embeddings."""

    # YaRN configuration (set via --yarn-* CLI args or --enable-gpt-oss preset)
    yarn_rotary_scaling_factor: Optional[float] = None
    """YaRN rotary scaling factor (e.g., 32.0 for GPT-OSS, 4.0 for Qwen)."""

    yarn_original_max_position_embeddings: Optional[int] = None
    """Original max position embeddings before YaRN scaling (e.g., 4096 for GPT-OSS)."""

    yarn_beta_fast: Optional[float] = None
    """YaRN beta_fast parameter for frequency interpolation (default: 32.0)."""

    yarn_beta_slow: Optional[float] = None
    """YaRN beta_slow parameter for frequency interpolation (default: 1.0)."""

    yarn_mscale: Optional[float] = None
    """YaRN mscale for attention scaling (default: 1.0)."""

    yarn_mscale_all_dim: Optional[float] = None
    """YaRN mscale_all_dim for attention scaling (default: 0.0)."""

    yarn_correction_range_round_to_int: bool = False
    """Whether to round correction range to integer in YaRN."""

    test_mode: bool = False
```

**검증 방법:**
```bash
grep -n "yarn_" megatron/core/transformer/transformer_config.py
# 출력: 196, 199, 202, 205, 208, 211, 214 라인에 yarn_* 필드 확인
```

---

### 3. `megatron/training/arguments.py`

#### 변경 1: Line 2236-2259 - YaRN CLI 인자 추가

**문제**: YaRN 파라미터를 CLI로 지정할 방법이 없음

**Before (Line 2216-2218):**
```python
    group.add_argument('--rope-type', type=str, default=None,
                      choices=['rope', 'yarn'],
                      help='Type of rope to use...')
    group.add_argument('--cross-entropy-loss-fusion', action='store_true',
```

**After (Line 2216-2260):**
```python
    group.add_argument('--rope-type', type=str, default=None,
                      choices=['rope', 'yarn'],
                      help='Type of rope to use...')
    group.add_argument('--enable-gpt-oss', action='store_true',
                      help='Enable GPT-OSS mode with YaRN RoPE configuration. When enabled, '
                      'automatically configures all YaRN parameters with GPT-OSS defaults '
                      '(yarn_original_max_position_embeddings=4096, factor=32, etc).')
    # YaRN RoPE configuration arguments (can override --enable-gpt-oss defaults)
    group.add_argument('--yarn-rotary-scaling-factor', type=float, default=None,
                      help='YaRN scaling factor (e.g., 32.0 for GPT-OSS, 4.0 for Qwen).')
    group.add_argument('--yarn-original-max-position-embeddings', type=int, default=None,
                      help='Original max position embeddings before YaRN scaling '
                      '(e.g., 4096 for GPT-OSS). This is NOT the extended length.')
    group.add_argument('--yarn-beta-fast', type=float, default=None,
                      help='YaRN beta_fast parameter (default: 32.0).')
    group.add_argument('--yarn-beta-slow', type=float, default=None,
                      help='YaRN beta_slow parameter (default: 1.0).')
    group.add_argument('--yarn-mscale', type=float, default=None,
                      help='YaRN mscale parameter (default: 1.0).')
    group.add_argument('--yarn-mscale-all-dim', type=float, default=None,
                      help='YaRN mscale_all_dim parameter (default: 0.0).')
    group.add_argument('--cross-entropy-loss-fusion', action='store_true',
```

**검증 방법:**
```bash
grep -n "yarn-" megatron/training/arguments.py
# 출력: --yarn-rotary-scaling-factor, --yarn-original-max-position-embeddings 등 확인
```

---

#### 변경 2: Line 1331-1349 - YaRN 설정 처리 로직

**문제**: CLI 인자를 config에 반영하는 로직 없음

**Before (Line 1325-1329):**
```python
        kw_args['quant_recipe'] = kitchen_quantization_recipe_config(args.kitchen_recipe_number)


    # Return config.
    return config_class(**kw_args)
```

**After (Line 1325-1349):**
```python
        kw_args['quant_recipe'] = kitchen_quantization_recipe_config(args.kitchen_recipe_number)


    # Create config
    config = config_class(**kw_args)

    # YaRN RoPE: --enable-gpt-oss preset or manual --yarn-* CLI args
    if getattr(args, 'enable_gpt_oss', False):
        config.position_embedding_type = "yarn"
        gpt_oss_defaults = {
            'yarn_rotary_scaling_factor': 32.0, 'yarn_original_max_position_embeddings': 4096,
            'yarn_beta_fast': 32.0, 'yarn_beta_slow': 1.0, 'yarn_mscale': 1.0, 'yarn_mscale_all_dim': 0.0,
        }
        for k, v in gpt_oss_defaults.items():
            if getattr(config, k, None) is None:
                setattr(config, k, v)

    # CLI args override config values
    for yarn_arg in ['yarn_rotary_scaling_factor', 'yarn_original_max_position_embeddings',
                     'yarn_beta_fast', 'yarn_beta_slow', 'yarn_mscale', 'yarn_mscale_all_dim']:
        val = getattr(args, yarn_arg, None)
        if val is not None:
            setattr(config, yarn_arg, val)

    return config
```

**검증 방법:**
```bash
grep -A 20 "# YaRN RoPE:" megatron/training/arguments.py
# gpt_oss_defaults 딕셔너리와 루프 패턴 확인
```

---

### 4. `megatron/post_training/model_builder.py`

#### 변경: Line 149-164 - 중복 YaRN 코드 제거

**문제**: `core_transformer_config_from_args`와 중복되는 YaRN 설정 코드

**Before (Line 146-164):**
```python
    print_rank_0("building GPT model ...")

    # ModelOpt by default assumes none homogenous layers...
    config = core_transformer_config_from_args(args)

    # Handle GPT-OSS mode with YaRN RoPE configuration
    if hasattr(args, 'enable_gpt_oss') and args.enable_gpt_oss:
        print_rank_0("GPT-OSS mode enabled: Configuring YaRN RoPE parameters")

        # Set GPT-OSS YaRN values directly on the config
        # These defaults are based on Huggingface GPT-OSS configurations
        config.position_embedding_type = "yarn"
        config.yarn_rotary_scaling_factor = 32.0
        config.yarn_original_max_position_embeddings = 131072  # ❌ 버그: 4096이어야 함
        config.yarn_beta_fast = 32.0
        config.yarn_beta_slow = 1.0
        config.yarn_mscale = 1.0
        config.yarn_mscale_all_dim = 0.0
        config.yarn_correction_range_round_to_int = False

    if vp_stage is not None:
```

**After (Line 146-152):**
```python
    print_rank_0("building GPT model ...")

    # ModelOpt by default assumes none homogenous layers...
    # NOTE: YaRN configuration (--enable-gpt-oss, --yarn-* args) is handled in core_transformer_config_from_args
    config = core_transformer_config_from_args(args)

    if vp_stage is not None:
```

**검증 방법:**
```bash
grep -A 5 "building GPT model" megatron/post_training/model_builder.py
# "NOTE: YaRN configuration" 주석 확인, 중복 코드 없음 확인
```

---

## mbridge 패치 상세

### `mbridge/models/gpt_oss.py`

**문제**: `yarn_original_max_position_embeddings` 기본값이 131072 (확장 후 길이)로 잘못 설정됨

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

**검증 방법:**
```bash
MBRIDGE_GPT_OSS=$(python3 -c "import mbridge.models.gpt_oss as m; print(m.__file__)")
grep "yarn_original_max_position_embeddings" "$MBRIDGE_GPT_OSS"
# 출력: yarn_original_max_position_embeddings: int = 4096
```

---

## SLIME 패치 상세

### 1. `slime/backends/megatron_utils/arguments.py`

**문제**: `max_position_embeddings`가 항상 `seq_length`로 덮어써짐

**Before:**
```python
def set_default_megatron_args(args):
    if not hasattr(args, 'seq_length') or args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length  # ❌ 항상 덮어씀
```

**After:**
```python
def set_default_megatron_args(args):
    if not hasattr(args, 'seq_length') or args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = getattr(args, "max_position_embeddings", None) or args.seq_length  # ✅ CLI 값 유지
```

**검증 방법:**
```bash
grep "max_position_embeddings" /path/to/slime/slime/backends/megatron_utils/arguments.py
# 출력: getattr(args, "max_position_embeddings", None) or args.seq_length
```

---

### 2. `slime/backends/megatron_utils/model_provider.py`

**문제**: Megatron v0.15.0이 전달하는 `config=`, `pg_collection=` kwargs 처리 불가

**Before:**
```python
def model_provider(pre_process=True, post_process=True, vp_stage=None):  # ❌ v0.15.0 비호환
```

**After:**
```python
def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):  # ✅ v0.15.0 호환
```

**검증 방법:**
```bash
grep "def model_provider" /path/to/slime/slime/backends/megatron_utils/model_provider.py
# 출력: def model_provider(pre_process=True, post_process=True, vp_stage=None, **kwargs):
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
