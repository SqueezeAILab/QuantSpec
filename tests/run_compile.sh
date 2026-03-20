#!/bin/bash
# Run QuantSpec with torch.compile and weight quantization
# Usage: ./run_compile.sh
# Modify the variables below to configure the run.

# --- Configuration ---
# Model checkpoint (converted with convert_hf_checkpoint.py)
model="checkpoints/meta-llama/Llama-2-7b-hf/model.pth"
model_name="meta-llama/Llama-2-7b-hf"
dataset="pg19"
marlin_path="marlin/gptq/checkpoint.pt.marlin.g128"

for prefix_len in 4000 8000 16000; do
  for gamma in 6; do
    if [ $prefix_len -gt 32000 ]; then
      gpus="0,1"
    else
      gpus="0"
    fi

    echo "----------------------------------------"
    echo "QuantSpec (compiled, weight quantized)"
    echo "Model: $model_name"
    echo "Prefix length: $prefix_len, Gamma: $gamma"
    echo "----------------------------------------"
    PYTHONWARNINGS=ignore CUDA_VISIBLE_DEVICES=$gpus ENABLE_INTRA_NODE_COMM=1 torchrun \
      --standalone \
      --nproc_per_node=$(echo $gpus | tr ',' ' ' | wc -w) \
      tests/quantspec_benchmark.py \
      --model $model \
      --model_name $model_name \
      --marlin_path $marlin_path \
      --dataset $dataset \
      --rank_group $(seq -s ' ' 0 $(($(echo $gpus | tr ',' ' ' | wc -w) - 1))) \
      --gamma $gamma \
      --B 1 \
      --prefix_len $prefix_len \
      --gen_len 90 \
      --printoutput \
      --wq \
      --benchmark \
      --compile
  done
done
