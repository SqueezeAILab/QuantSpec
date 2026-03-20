#!/bin/bash
# Example: Run sensitivity analysis and speculative decoding evaluation
# Usage: ./dynamic.sh <cuda_device_number>

if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# Step 1: Compute per-layer sensitivity (Fisher information)
for prefill in 1024
do
    CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/testing_model.py \
        --model meta-llama/Llama-2-7b-hf \
        --k_bits 4 --v_bits 4 \
        --group_size 128 --residual_length 128 \
        --prefill $prefill \
        --simquant \
        --include_error \
        --filename "llama_2_7b_simquant_k4v4_gs128_rl0_sensitivity.json"
done
