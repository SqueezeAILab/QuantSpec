
if [ $# -eq 0 ]; then
    echo "Please provide the CUDA device number as an argument."
    echo "Usage: $0 <cuda_device_number>"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$1 python test/testing_model.py --prefill 2048 --verbose --gamma 6 --draft meta-llama/Meta-Llama-3.1-8B --target meta-llama/Meta-Llama-3.1-8B --temp 0.01