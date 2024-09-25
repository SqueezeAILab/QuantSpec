
for prefill in 512 1024 4096 16384 32768
do
    for k_bits in {2..4}
    do
        for group_size in 8 16 32 64 128
        do
            CUDA_VISIBLE_DEVICES=0 python test/on_chip.py --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6
        done
    done
done