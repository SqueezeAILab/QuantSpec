# QuantSpec (HuggingFace Implementation)

This branch contains an alternative HuggingFace-based implementation of QuantSpec using the `transformers` library.
It implements stochastic speculative decoding with rejection sampling and supports mixed-precision per-layer KV quantization with sensitivity analysis.

For the main implementation (with torch.compile support, weight quantization, and broader model support), see the `main` branch.

For details, see the paper: [QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache](https://arxiv.org/abs/2502.10424)
