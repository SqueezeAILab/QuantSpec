python test/on_chip.py \
    --prefill 4096 \
    --target NousResearch/Yarn-Llama-2-7b-128k \
    --draft NousResearch/Yarn-Llama-2-7b-128k \
    --verbose \
    --gamma 4 \
    --k_bits -16 \
    --v_bits -16 \
    --temp 0.001 \
    --cache_dir /rscratch/xihc/.cache/

# 
    # --quantized_draft \