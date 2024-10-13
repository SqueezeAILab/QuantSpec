
for prefill in 2048 4096 16384 32768
do
    for k_bits in 2 4
    do
        for group_size in 16
        do
            CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python test/on_chip.py --simquant --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 --identifier_string "llama 3.1 8b simqant matmulfp16"
        done
    done
done