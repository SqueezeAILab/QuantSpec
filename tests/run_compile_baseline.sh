#!/bin/bash
# Run autoregressive baseline benchmark (no speculative decoding)
# Usage: ./run_compile_baseline.sh <cuda_device(s)> <prefix_len>
# Example: ./run_compile_baseline.sh 0 4000

if [ $# -lt 2 ]; then
    echo "Usage: $0 <cuda_device(s)> <prefix_len>"
    exit 1
fi

PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
    --standalone \
    --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
    tests/baseline_benchmark.py \
    --model checkpoints/meta-llama/Llama-2-7b-hf/model.pth \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset pg19 \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --B 1 \
    --prefix_len $2 \
    --gen_len 90 \
    --compile
