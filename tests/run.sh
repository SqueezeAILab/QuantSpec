#!/bin/bash
# Run QuantSpec speculative decoding benchmark
# Usage: ./run.sh <cuda_device(s)>
# Example: ./run.sh 0
#          ./run.sh 0,1  (for tensor parallel)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$1 ENABLE_INTRA_NODE_COMM=1 torchrun \
    --standalone \
    --nproc_per_node=$(echo $1 | tr ',' ' ' | wc -w) \
    tests/quantspec_benchmark.py \
    --model checkpoints/meta-llama/Llama-2-7b-hf/model.pth \
    --model_name meta-llama/Llama-2-7b-hf \
    --marlin_path marlin/gptq/checkpoint.pt.marlin.g128 \
    --rank_group $(seq -s ' ' 0 $(($(echo $1 | tr ',' ' ' | wc -w) - 1))) \
    --gamma 6 \
    --B 1 \
    --prefix_len 1024 \
    --gen_len 64 \
    --benchmark \
    --compile
