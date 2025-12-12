#!/bin/bash
# YaRN General Patch Application Script
# Supports GPT-OSS, Qwen, and other YaRN models
#
# Usage: ./apply_yarn_patches.sh [options]
#   --megatron-only    Only apply Megatron patch
#   --slime-only       Only apply SLIME patch
#   --mbridge-only     Only apply mbridge patch
#   --dry-run          Show what would be done without applying

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=false
APPLY_MEGATRON=true
APPLY_SLIME=true
APPLY_MBRIDGE=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --megatron-only)
            APPLY_SLIME=false
            APPLY_MBRIDGE=false
            shift
            ;;
        --slime-only)
            APPLY_MEGATRON=false
            APPLY_MBRIDGE=false
            shift
            ;;
        --mbridge-only)
            APPLY_MEGATRON=false
            APPLY_SLIME=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "YaRN General Patch Application"
echo "============================================"
echo ""

# Apply Megatron patch
if [ "$APPLY_MEGATRON" = true ]; then
    echo "[1/3] Megatron-LM patch"
    MEGATRON_DIR="${MEGATRON_DIR:-$(dirname "$SCRIPT_DIR")}"
    if [ -d "$MEGATRON_DIR/megatron" ]; then
        echo "  Directory: $MEGATRON_DIR"
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would apply: megatron_yarn_general.patch"
        else
            cd "$MEGATRON_DIR"
            patch -p1 < "$SCRIPT_DIR/megatron_yarn_general.patch"
            echo "  ✓ Applied successfully"
        fi
    else
        echo "  ✗ Megatron directory not found: $MEGATRON_DIR"
        echo "  Set MEGATRON_DIR environment variable"
    fi
    echo ""
fi

# Apply SLIME patch
if [ "$APPLY_SLIME" = true ]; then
    echo "[2/3] SLIME patch"
    SLIME_DIR="${SLIME_DIR:-}"
    if [ -z "$SLIME_DIR" ]; then
        # Try to find SLIME
        SLIME_DIR=$(python3 -c "import slime; import os; print(os.path.dirname(slime.__file__))" 2>/dev/null || echo "")
    fi
    if [ -n "$SLIME_DIR" ] && [ -d "$SLIME_DIR" ]; then
        echo "  Directory: $SLIME_DIR"
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would apply: slime_yarn_general.patch"
        else
            cd "$SLIME_DIR/.."
            patch -p1 < "$SCRIPT_DIR/slime_yarn_general.patch" || echo "  ⚠ Patch may already be applied or needs manual review"
            echo "  ✓ Applied (check for errors above)"
        fi
    else
        echo "  ✗ SLIME not found. Set SLIME_DIR environment variable"
        echo "  Example: export SLIME_DIR=/path/to/slime"
    fi
    echo ""
fi

# Apply mbridge patch
if [ "$APPLY_MBRIDGE" = true ]; then
    echo "[3/3] mbridge patch"
    MBRIDGE_DIR="${MBRIDGE_DIR:-}"
    if [ -z "$MBRIDGE_DIR" ]; then
        # Try to find mbridge
        MBRIDGE_DIR=$(python3 -c "import mbridge; import os; print(os.path.dirname(mbridge.__file__))" 2>/dev/null || echo "")
    fi
    if [ -n "$MBRIDGE_DIR" ] && [ -d "$MBRIDGE_DIR" ]; then
        echo "  Directory: $MBRIDGE_DIR"
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would apply: mbridge_yarn_fix.patch"
        else
            cd "$MBRIDGE_DIR/.."
            patch -p1 < "$SCRIPT_DIR/mbridge_yarn_fix.patch" || echo "  ⚠ Patch may already be applied or needs manual review"
            echo "  ✓ Applied (check for errors above)"
        fi
    else
        echo "  ✗ mbridge not found. Set MBRIDGE_DIR environment variable"
        echo "  Or install: pip install mbridge"
        echo ""
        echo "  Manual fix (if patch fails):"
        echo "  Edit mbridge/models/gpt_oss.py and change:"
        echo "    yarn_original_max_position_embeddings: int = 131072"
        echo "  To:"
        echo "    yarn_original_max_position_embeddings: int = 4096"
    fi
    echo ""
fi

echo "============================================"
echo "Done!"
echo ""
echo "Usage examples after patching:"
echo ""
echo "  # GPT-OSS (preset)"
echo "  --enable-gpt-oss --max-position-embeddings 131072"
echo ""
echo "  # GPT-OSS (manual)"
echo "  --position-embedding-type yarn \\"
echo "  --yarn-original-max-position-embeddings 4096 \\"
echo "  --yarn-rotary-scaling-factor 32.0 \\"
echo "  --max-position-embeddings 131072"
echo ""
echo "  # Qwen (example)"
echo "  --position-embedding-type yarn \\"
echo "  --yarn-original-max-position-embeddings 32768 \\"
echo "  --yarn-rotary-scaling-factor 4.0 \\"
echo "  --max-position-embeddings 131072"
echo "============================================"
