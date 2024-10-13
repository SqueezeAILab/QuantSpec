for prefill in 2048 4096 16384 32768
do
    for k_bits in 2 4
    do
        for group_size in 16
        do
            CUDA_VISIBLE_DEVICES=6 TOKENIZERS_PARALLELISM=false python test/on_chip.py --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --temp 0.0 --top_p 0.0 --identifier_string "llama 3.1 8b actual quantization" --log_file "on_chip_results1.txt"
        done
    done
done