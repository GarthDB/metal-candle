# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-TBD

### Added

#### Core Features
- **LoRA Training Pipeline**: Complete Low-Rank Adaptation implementation for efficient fine-tuning
  - LoRA layers with configurable rank and alpha parameters
  - Support for Q-Proj, K-Proj, V-Proj, and O-Proj target modules
  - Gradient flow verification and backpropagation support
  - **Performance**: 1.5-2.4x faster than MLX for LoRA operations

- **Model Loading & Architecture**:
  - Safetensors format support with validation
  - Qwen2.5-Coder architecture implementation
  - Transformer components: RoPE embeddings, multi-head attention (GQA), MLP layers
  - Model configuration from JSON files
  - Builder pattern API with sensible defaults

- **Training Infrastructure**:
  - AdamW optimizer with decoupled weight decay
  - Learning rate schedulers: Constant, Linear, Cosine, WarmupCosine
  - Cross-entropy loss with optional label smoothing
  - Gradient clipping and accumulation
  - Checkpoint management (save/load with metadata)

- **Inference & Text Generation**:
  - KV-cache for efficient token generation (~173 MB for 2048 tokens, Qwen 0.5B F16)
  - Multiple sampling strategies: Greedy, Top-k, Top-p (nucleus), Temperature
  - Memory-efficient O(1) position tracking
  - Sampling overhead <1% of forward pass time

- **Semantic Embeddings** (feature: `embeddings`):
  - Sentence-transformers support: E5-small-v2, MiniLM-L6-v2, MPNet-base-v2
  - HuggingFace Hub integration with auto-download and caching
  - Mean pooling with attention weighting
  - L2 normalization for cosine similarity
  - Works on both CPU and Metal devices

- **Metal Acceleration**:
  - Native Apple Silicon Metal backend via Candle
  - Optimized matrix operations for LoRA
  - 2-5x speedup for layer operations (softmax, layer norm, RMS norm)
  - Efficient GPU utilization for high-rank LoRA operations

#### Quality & Documentation
- **160 comprehensive tests**: 144 unit tests + 6 gradient tests + 10 inference tests + 43 doctests
- **Zero clippy warnings**: Strict pedantic linting enforced
- **84.69% code coverage**: Exceeds 80% requirement
- **100% API documentation**: All public APIs fully documented with examples
- **6 working examples**: Demonstrating all major features
- **Complete architecture documentation**: ARCHITECTURE.md, CONTRIBUTING.md, BENCHMARKS.md

#### Performance
- **LoRA Operations**: 149-244% of MLX performance (1.49-2.44x faster) 🚀
- **Overall Performance**: 110% of MLX baseline
- **KV-Cache**: Minimal overhead, <1% of generation time
- **Metal GPU**: 2-5x speedup over CPU for tensor operations

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- No known security vulnerabilities
- All dependencies audited with `cargo deny`
- Two unmaintained transitive dependencies (not security issues):
  - `number_prefix` (via hf-hub → indicatif)
  - `paste` (via candle-core → gemm/metal)
  - Both from trusted upstream, will be resolved when dependencies update

## Performance Benchmarks

Detailed benchmarks available in [BENCHMARKS.md](BENCHMARKS.md).

### LoRA Operations (vs MLX)
- Small (512×512, rank=8): **1.49x faster**
- Medium (1024×1024, rank=8): **1.57x faster**
- Large (2048×2048, rank=8): **2.44x faster**

### Metal GPU Acceleration
- LoRA forward pass: 1.76-3.14x faster than CPU
- Softmax: 5.21x faster than CPU
- Layer Norm: 2.53x faster than CPU
- RMS Norm: 2.42x faster than CPU

## Known Limitations

### v1.0.0 Limitations
- **Model Format**: Safetensors only (GGUF planned for v1.1+)
- **Model Architecture**: Qwen2.5-Coder only (more architectures planned)
- **Apple Silicon Only**: Requires M1/M2/M3/M4 chip with Metal support
- **Layer Operations**: Transformer operations (softmax, layer norm) slower than MLX
  - Impact: Minimal for LoRA training (not in training loop)
  - Planned: Custom Metal kernels for v1.1+ if needed
- **Single GPU**: Multi-GPU support planned for v2.0

### Recommendations
- **Best for**: LoRA training and fine-tuning (1.5-2.4x faster than MLX)
- **Good for**: Inference with LoRA adapters
- **Consider MLX for**: Full transformer inference without LoRA

## Upgrading

This is the initial v1.0.0 release. No upgrade path needed.

## Migration from MLX+PyO3

For users migrating from Ferris project's MLX+PyO3 implementation:

1. **Remove Python dependencies**: No Python runtime or virtual environment needed
2. **Update model loading**: Use `ModelLoader` builder API
3. **Update LoRA training**: Use `LoRAAdapter` and `Trainer` APIs
4. **Performance**: Expect 1.5-2.4x speedup for LoRA operations
5. **Deployment**: Single binary, no Python packaging needed

See migration guide in documentation for detailed steps.

## Future Roadmap

### v1.1 (Planned)
- GGUF format support
- Additional model architectures (LLaMA, Mistral)
- Optional transformer component optimization
- Advanced LoRA variants (DoRA, etc.)

### v1.2+ (Planned)
- Quantization support (4-bit, 8-bit)
- Flash Attention integration
- Streaming generation with callbacks
- Batched inference optimization

### v2.0 (Planned)
- Multi-GPU training support
- Custom Metal kernel implementations
- Model quantization and compression

## Contributors

- [@GarthDB](https://github.com/GarthDB) - Initial implementation and design

## Acknowledgments

- Built on the excellent [Candle](https://github.com/huggingface/candle) framework by Hugging Face
- Inspired by [MLX](https://github.com/ml-explore/mlx) and [llama.cpp](https://github.com/ggerganov/llama.cpp)
- LoRA implementation based on [LoRA paper](https://arxiv.org/abs/2106.09685) by Hu et al.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

**Status**: ✅ Production Ready  
**Target Platform**: Apple Silicon (M1/M2/M3/M4)  
**Minimum Requirements**: Rust 1.75+, macOS 12.0+

[1.0.0]: https://github.com/GarthDB/metal-candle/releases/tag/v1.0.0

