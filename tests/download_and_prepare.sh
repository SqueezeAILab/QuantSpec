#!/bin/bash
# Download and convert a HuggingFace model for QuantSpec
# Usage: ./download_and_prepare.sh <model_repo_id> [--hf_token <token>] [--out_dir <dir>]
#
# Example:
#   ./download_and_prepare.sh meta-llama/Llama-2-7b-hf --hf_token hf_xxx --out_dir ./checkpoints

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_repo_id> [--hf_token <token>] [--out_dir <dir>]"
    echo "Example: $0 meta-llama/Llama-2-7b-hf --hf_token hf_xxx"
    exit 1
fi

MODEL_ID=$1
shift

# Default output directory
OUT_DIR="./checkpoints"
HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf_token) HF_TOKEN=$2; shift 2 ;;
        --out_dir) OUT_DIR=$2; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

EXTRA_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    EXTRA_ARGS="--hf_token $HF_TOKEN"
fi

python download.py \
    --repo_id $MODEL_ID \
    --out_dir $OUT_DIR/$MODEL_ID \
    $EXTRA_ARGS

python convert_hf_checkpoint.py \
    --checkpoint_dir $OUT_DIR/$MODEL_ID
