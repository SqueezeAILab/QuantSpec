if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/testing_model.py \
#     --attn flash_attention_2 \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --simquant \
#     --k_bits 2 \
#     --v_bits 2 \
#     --group_size 16 \
#     --residual_length 16 \
#     --prefill 2048

CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/testing_model.py \
    --attn flash_attention_2 \
    --model meta-llama/Meta-Llama-3.1-8B \
    --simquant \
    --quantized \
    --k_bits 2 \
    --v_bits 2 \
    --group_size 16 \
    --residual_length 0 \
    --prefill 1024 \
    --num_examples 20 \
    --filename "llama_3.1_8b_simquant_k2v2_gs16_rl0_grads.json"


# CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/testing_model.py \
#     --attn flash_attention_2 \
#     --model meta-llama/Meta-Llama-3.1-8B \
#     --simquant \
#     --quantized \
#     --k_bits 8 \
#     --v_bits 8 \
#     --group_size 16 \
#     --residual_length 16 \
#     --prefill 2048