
if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$1 python test/on_chip.py --prefill 128 --quantized_draft --verbose --gamma 6 --nuq --draft /rscratch/llama-7b/ --target /rscratch/llama-7b/