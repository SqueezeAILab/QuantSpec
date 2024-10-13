
if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# CUDA_VISIBLE_DEVICES=$1 python test/testing_model.py --prefill 1024 --verbose --gamma 6 --draft meta-llama/Llama-2-7b-hf --target meta-llama/Llama-2-7b-hf --temp 0.01 --quantized_draft --k_bits 2 --v_bits 2 --group_size 32 --residual_length 32 --identifier_string "testing kivi"
CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/on_chip.py --verbose --top_p 0.0 --temp 0.0 --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits 2 --v_bits 2 --group_size 16 --residual_length 128 --prefill 2048 --gen_len 64 --gamma 6 --identifier_string "testing kivi"
