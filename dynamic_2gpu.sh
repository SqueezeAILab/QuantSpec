if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

for prefill in 8192
do
    CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/spec.py --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --k_bits 4 --v_bits 4 --group_size 128 --residual_length 128 --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 \
    --prefill $prefill \
    --simquant \
    --identifier_string "llama 3.1 8b simqant dynamic gs 0" \
    --first_few_fp16_layers 8
done