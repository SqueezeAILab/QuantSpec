if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

# --simquant \

# target fp16, draft int8_with_upper_lower_bits
for prefill in 2048
do
    CUDA_VISIBLE_DEVICES=$1 TOKENIZERS_PARALLELISM=false python test/on_chip.py --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits 4 --v_bits 4 --group_size 128 --residual_length 128 --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 \
    --prefill $prefill \
    --mixed_precision \
    --sensitivity_filepath /rscratch/rishabhtiwari/QuantSpec/results/llama_3.1_8b_simquant_k2v2_gs16_rl0_sensitivity.json \
    --percentile 0.05 \
    --identifier_string "llama 3.1 8b simqant dynamic gs 0"
done




# for threshold in 0.05 0.1 0.2 0.3 0.4 0.5 0.95
# do 
#     for prefill in 2048
#     do
#         for k_bits in 8
#         do
#             for group_size in 16
#             do
#                 CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python test/dynamic.py --simquant --per_layer_threshold --fisher_file_path /rscratch/rishabhtiwari/QuantSpec/results/llama_3.1_8b_simquant_k2v2_gs16_rl0_sensitivity.json --percentile $threshold --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 --identifier_string "llama 3.1 8b simqant dynamic gs $threshold"
#             done
#         done
#     done

#     for prefill in 2048
#     do
#         for k_bits in 8
#         do
#             for group_size in 16
#             do
#                 CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python test/dynamic.py --simquant --global_threshold --per_layer_threshold --fisher_file_path /rscratch/rishabhtiwari/QuantSpec/results/llama_3.1_8b_simquant_k2v2_gs16_rl0_sensitivity.json --percentile $threshold --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 --identifier_string "llama 3.1 8b simqant dynamic gs $threshold"
#             done
#         done
#     done

#     for prefill in 2048
#     do
#         for k_bits in 8
#         do
#             for group_size in 16
#             do
#                 CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python test/dynamic.py --simquant --fisher_file_path /rscratch/rishabhtiwari/QuantSpec/results/llama_3.1_8b_simquant_k2v2_gs16_rl0_sensitivity.json --percentile $threshold --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 --identifier_string "llama 3.1 8b simqant dynamic gs $threshold"
#             done
#         done
#     done

#     for prefill in 2048
#     do
#         for k_bits in 8
#         do
#             for group_size in 16
#             do
#                 CUDA_VISIBLE_DEVICES=7 TOKENIZERS_PARALLELISM=false python test/dynamic.py --simquant --global_threshold --fisher_file_path /rscratch/rishabhtiwari/QuantSpec/results/llama_3.1_8b_simquant_k2v2_gs16_rl0_sensitivity.json --percentile $threshold --verbose --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --quantized_draft --k_bits $k_bits --v_bits $k_bits --group_size $group_size --residual_length 128 --prefill $prefill --gen_len 256 --gamma 6 --top_p 0.0 --temp 0.0 --identifier_string "llama 3.1 8b simqant dynamic gs $threshold"
#             done
#         done
#     done
# done