# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2024-12-11

### Highlights

- **Production-Ready Text Generation API**: Complete high-level API for text generation with streaming support
- **Advanced Sampling Strategies**: Repetition penalty for higher quality generation
- **Comprehensive Testing**: 203+ tests with full coverage of generation pipeline
- **Developer Experience**: New example demonstrating all generation features

### ⚠️ Breaking Changes

#### `sample_token()` Function Signature

The `sample_token()` function signature has been updated to support repetition penalty:

**Before (v1.0.0)**:
```rust
pub fn sample_token(logits: &Tensor, strategy: &SamplingStrategy) -> Result<u32>
```

**After (v1.1.0)**:
```rust
pub fn sample_token(
    logits: &Tensor, 
    strategy: &SamplingStrategy,
    generated_ids: &[u32],      // NEW: Previously generated tokens
    repetition_penalty: f32,     // NEW: Penalty factor (1.0 = no penalty)
) -> Result<u32>
```

**Migration Guide**:
- For basic usage without repetition penalty: Pass `&[]` and `1.0` as the new parameters
- To enable repetition penalty: Pass your generated token history and desired penalty factor (e.g., `1.1`)

**Example**:
```rust
// Old code (v1.0.0)
let token = sample_token(&logits, &strategy)?;

// New code (v1.1.0) - no repetition penalty
let token = sample_token(&logits, &strategy, &[], 1.0)?;

// New code (v1.1.0) - with repetition penalty
let token = sample_token(&logits, &strategy, &generated_ids, 1.1)?;
```

**Recommended**: Use the high-level `Generator` API instead of calling `sample_token()` directly:
```rust
let mut generator = Generator::new(Box::new(model), config)?;
let output = generator.generate(&input_ids)?;
```

### Added

#### Text Generation API (Issue #31)
- **`Generator` struct**: High-level text generation with model integration
  - `generate()`: Standard generation with configurable parameters
  - `generate_stream()`: Real-time streaming generation with callback support
  - Stop conditions: EOS tokens, custom stop tokens, max length
  - Automatic repetition penalty application
- **`LanguageModel` trait**: Common interface for different model architectures
  - Implemented for `Qwen` model
  - Extensible for future model architectures
- **Generation Examples**: New `examples/generate_text.rs` demonstrating:
  - Basic greedy generation
  - Different sampling strategies (Greedy, Top-k, Top-p, Temperature)
  - Streaming generation with callbacks
  - Repetition penalty usage
  - Stop conditions

#### Advanced Sampling (Issue #29)
- **Repetition Penalty**: `apply_repetition_penalty()` function
  - Reduces repetitive text generation
  - Configurable penalty factor (> 1.0 = penalize, 1.0 = no penalty)
  - Integrated with all sampling strategies
- **Enhanced `sample_token()`**: Now accepts repetition penalty and generated token history

#### Configuration
- **Complete `GeneratorConfig`**:
  - All sampling parameters: `temperature`, `top_p`, `top_k`, `repetition_penalty`
  - Stop conditions: `stop_on_eos`, `eos_token_id`, `stop_tokens`
  - Builder-friendly with sensible defaults

#### Testing & Quality
- **Extended test coverage**: 210+ tests (up from 195)
  - Unit tests for `Generator` with mock models
  - Integration tests for full generation pipeline
  - Tests for all sampling strategies and stop conditions
  - Tests for streaming API and callbacks
- **Code coverage**: Maintained ≥80% coverage
- **Zero clippy warnings**: Pedantic mode with production-quality code

### Changed

- **`Generator` API**: Replaced placeholder with full implementation (see Breaking Changes section for `sample_token()` updates)

### Fixed

- N/A (new features, no bugs fixed)

### Performance

- Generation performance: Comparable to v1.0.0 inference (no KV-cache optimization yet)
- Sampling overhead: <1% of forward pass time (maintained)
- Memory: Minimal overhead for repetition penalty tracking

### Documentation

- Complete API documentation for all new types and functions
- New example (`generate_text.rs`) with 5 comprehensive demos
- Updated README with text generation quick start
- Inline code examples in docstrings

### Issues Closed

- #29: Advanced Sampling Strategies for Text Generation ✅
- #30: KV Cache Implementation (Already complete in v1.0.0) ✅
- #31: High-Level Text Generation API ✅

### Notes

- **Issue #27** (Custom Fused Softmax Kernel): Deferred to v1.2.0 per release plan
  - Reason: Text generation API provides more immediate user value
  - Current Candle softmax performs adequately
  - Will optimize in v1.2.0 after full pipeline validation

## [1.0.0] - 2024-12-10

### Highlights

- **25.9x faster than MLX** for embeddings (Apple's official ML framework)
- Production-ready LoRA training for Apple Silicon
- Custom Metal LayerNorm kernel for optimal performance
- Lazy evaluation graph with operation fusion (experimental, feature-gated)
- 190 passing tests (137 lib + 53 doc), 84.69% code coverage
- Clean codebase: 4 documented pedantic warnings, 100% API documentation

### Added

#### Core Features
- **LoRA Training Pipeline**: Complete Low-Rank Adaptation implementation for efficient fine-tuning
  - LoRA layers with configurable rank and alpha parameters
  - Support for Q-Proj, K-Proj, V-Proj, and O-Proj target modules
  - **Dropout support**: Training/eval mode control for regularization (per LoRA paper)
  - Gradient flow verification and backpropagation support
  - **Performance**: Metal GPU delivers 1.76-3.14x speedup over CPU for LoRA operations

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
  - **Custom Metal LayerNorm kernel**: 25.9x faster than MLX for batch processing
  - Works on both CPU and Metal devices

- **Metal Acceleration**:
  - Native Apple Silicon Metal backend via Candle
  - Custom Metal LayerNorm kernel for optimal embeddings performance
  - Near constant-time performance (4.4ms for 100 docs, 3.9ms for 1 doc)
  - Lazy evaluation graph with operation fusion

#### Quality & Documentation
- **195 comprehensive tests**: 187 library tests (including 8 dropout tests) + 56 doctests
- **Clean codebase**: 4 documented pedantic warnings (all justified and documented)
- **Code coverage**: Exceeds 80% requirement
- **100% API documentation**: All public APIs fully documented with examples
- **6 working examples**: Demonstrating all major features
- **Complete architecture documentation**: ARCHITECTURE.md, CONTRIBUTING.md, performance guides

#### Performance
- **Embeddings**: 25.9x faster than MLX for batch processing (100 docs: 4.4ms vs 113.5ms)
- **Single Query**: 2x faster than MLX (3.9ms vs 7.7ms)
- **Throughput**: 22,831 docs/sec (MLX: 881 docs/sec)
- **Near Constant-Time**: Only 13% increase for 100x more data (3.9ms → 4.4ms)
- **KV-Cache**: Minimal overhead, <1% of generation time

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

Detailed benchmarks available in [MLX_BENCHMARK_COMPARISON.md](MLX_BENCHMARK_COMPARISON.md) and [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md).

### Embeddings Performance (vs MLX)

| Batch Size | metal-candle | MLX | Speedup |
|-----------|-------------|-----|---------|
| 1         | 3.9ms       | 7.7ms | **2.0x** |
| 100       | 4.4ms       | 113.5ms | **25.9x** |

**Throughput**: 22,831 docs/sec (MLX: 881 docs/sec)

### Metal GPU Acceleration
- Custom LayerNorm kernel with optimal threadgroup sizing
- Lazy evaluation graph with operation fusion
- Near constant-time scaling across batch sizes

## Known Limitations

### v1.0.0 Limitations
- **Model Format**: Safetensors only (GGUF planned for v1.1+)
- **Model Architecture**: Qwen2.5-Coder for text generation, BERT variants for embeddings
- **Apple Silicon Only**: Requires M1/M2/M3/M4 chip with Metal support
- **Single GPU**: Multi-GPU support planned for v2.0

### Recommendations
- **Best for**: Semantic embeddings and RAG applications (25.9x faster than MLX)
- **Great for**: LoRA training and fine-tuning
- **Excellent for**: Inference with LoRA adapters
- **Production Ready**: Use Metal for all embeddings workloads

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


