#!/bin/bash
# Example: Run speculative decoding with mixed-precision KV cache on 2 GPUs
# Usage: ./dynamic_2gpu.sh <cuda_device_number>

if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# Run speculative decoding with mixed-precision quantization
for prefill in 2048
do
    for temp in 0.2 0.4 0.6 0.8 1
    do
        for percentile in 0.1
        do
            CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/spec.py \
                --draft meta-llama/Llama-2-7b-hf \
                --target meta-llama/Llama-2-7b-hf \
                --k_bits 4 --v_bits 4 \
                --group_size 128 --residual_length 128 \
                --gen_len 256 --gamma 6 --temp $temp \
                --prefill $prefill \
                --dataset gs \
                --mixed_precision \
                --percentile $percentile \
                --sensitivity_filepath "./test/llama_2_7b_simquant_k4v4_gs128_rl0_sensitivity.json" \
                --identifier_string "llama2_7b_mixed_precision"
        done
    done
done
