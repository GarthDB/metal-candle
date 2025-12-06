# Frequently Asked Questions

Common questions about metal-candle.

## General

### What is metal-candle?

metal-candle is a production-quality Rust library for machine learning on Apple Silicon, specializing in LoRA training, model loading, and text generation using the Candle framework with Metal backend.

### Why use metal-candle instead of PyTorch/MLX?

**Advantages:**
- **Pure Rust**: No Python runtime needed, single binary deployment
- **Type Safety**: Compile-time error checking prevents entire classes of bugs
- **Memory Safety**: Rust's ownership system prevents segfaults and data races
- **Simple Deployment**: No virtual environments or Python dependencies
- **Production Quality**: 160 tests, zero warnings, ≥80% code coverage

**Trade-offs:**
- Smaller ecosystem than PyTorch/MLX
- Only supports Apple Silicon (not cross-platform)
- Raw performance currently optimized for ergonomics over speed

### Is metal-candle production-ready?

Yes! Version 1.0.0 has:
- 160 passing tests
- 84.69% code coverage (exceeds 80% requirement)
- Zero clippy warnings (pedantic level)
- 100% API documentation
- Metal GPU acceleration for Apple Silicon

## Installation & Setup

### Do I need an Apple Silicon Mac?

Yes. metal-candle is specifically designed for Apple Silicon (M1/M2/M3/M4) and requires Metal GPU support.

### What macOS version do I need?

macOS 12.0 (Monterey) or later for Metal support.

### Can I use this on Intel Mac / Linux / Windows?

No. metal-candle requires Apple's Metal framework which is only available on Apple Silicon Macs. For other platforms, consider [Candle](https://github.com/huggingface/candle) or [PyTorch](https://pytorch.org).

### Why is my first build so slow?

The first build compiles Candle and all dependencies, which can take 5-10 minutes. Subsequent builds will be much faster (seconds).

## Performance

### How does metal-candle perform?

Metal GPU provides significant speedup over CPU:
- Small (512×512, r=8): 1.76x faster
- Medium (1024×1024, r=8): 2.29x faster  
- Large (2048×2048, r=8): 2.67x faster

**Value Proposition**: While raw throughput is optimized for correctness, metal-candle excels in type safety, ergonomic APIs, and single-binary deployment.

See [Benchmarks](../testing/benchmarks.md) for complete metrics.

### What are metal-candle's strengths?

- **Type Safety**: Compile-time guarantees prevent bugs
- **Single Binary**: No Python runtime or virtual environments
- **Memory Safety**: No segfaults or data races
- **Production Quality**: Comprehensive tests and documentation
- **Ergonomic APIs**: Builder patterns and clear error messages

### Should I use GPU or CPU?

**Use Metal GPU for:**
- Model forward passes
- Large tensor operations (>1000 elements)
- Training

**Use CPU for:**
- Sampling/token selection
- Small operations (<1000 elements)
- Testing/debugging

See [Device Management](../guide/devices.md).

## Models & Training

### What models are supported?

**v1.0:**
- Qwen2.5-Coder (all sizes: 0.5B, 1.5B, 3B, 7B)

**Future (v1.1+):**
- LLaMA, Mistral, and other architectures

### What model formats are supported?

**v1.0:**
- Safetensors (primary)

**Future (v1.1+):**
- GGUF (llama.cpp compatibility)
- PyTorch (legacy support)

### How much memory do I need?

Depends on model size and precision:

| Model | F16 Memory | F32 Memory |
|-------|------------|------------|
| 0.5B | ~1 GB | ~2 GB |
| 1.5B | ~3 GB | ~6 GB |
| 3B | ~6 GB | ~12 GB |
| 7B | ~14 GB | ~28 GB |

Recommended: 16GB+ unified memory for 7B models.

### Can I fine-tune the entire model?

metal-candle specializes in LoRA (parameter-efficient fine-tuning). Full fine-tuning is not currently supported but may be added in future versions.

### What's the best LoRA rank to use?

Start with **rank=8** (good balance). Increase to 16 or 32 if you need more capacity. Higher rank = more parameters but diminishing returns.

## Technical

### Why F16 instead of F32?

F16 (half precision):
- Uses half the memory
- Faster on Metal GPU
- Sufficient precision for most ML tasks

Use F32 only when you need higher precision.

### What's "contiguous" and why does it matter?

After operations like `transpose()`, tensors may not be contiguous in memory. Metal operations require contiguous tensors:

```rust
// Make contiguous after transpose
let t = tensor.transpose(0, 1)?.contiguous()?;
```

### Can I use multiple GPUs?

Not in v1.0. Multi-GPU support is planned for v2.0.

### Does it support quantization?

Not in v1.0. Quantization (4-bit, 8-bit) is planned for v1.1+.

## Errors & Troubleshooting

### "Metal device not available"

- Ensure you're on Apple Silicon (not Intel Mac)
- Check macOS version ≥12.0
- Restart your Mac

### "Shape mismatch" errors

Check that tensor shapes are compatible for the operation. For matrix multiplication:
```
(m, n) × (n, p) = (m, p)
```

### "Unexpected striding" errors

Make tensors contiguous after reshape/transpose:
```rust
let t = tensor.transpose(0, 1)?.contiguous()?;
```

See [Troubleshooting Guide](./troubleshooting.md) for more.

## Development

### How do I contribute?

See [Contributing Guide](../development/contributing.md) for:
- Code quality standards
- Testing requirements
- PR process

### Is there a roadmap?

Yes! See [Roadmap](../development/roadmap.md) for planned features:
- v1.1: GGUF support, more model architectures
- v1.2+: Quantization, Flash Attention
- v2.0: Multi-GPU, custom Metal kernels

### Can I use this in commercial projects?

Yes! metal-candle is licensed under Apache-2.0, which allows commercial use.

## Getting Help

### Where can I get help?

- [GitHub Issues](https://github.com/GarthDB/metal-candle/issues) - Bug reports, feature requests
- [Troubleshooting Guide](./troubleshooting.md) - Common problems
- [API Documentation](https://docs.rs/metal-candle) - Complete reference

### How do I report a bug?

Open an issue on [GitHub](https://github.com/GarthDB/metal-candle/issues) with:
- Rust version (`rustc --version`)
- macOS version
- Minimal code to reproduce
- Error messages

### Can I request a feature?

Yes! Open an issue describing:
- Use case
- Expected behavior
- Why it's valuable

## Comparisons

### metal-candle vs Candle

- **metal-candle**: Higher-level abstractions for LoRA training
- **Candle**: Lower-level ML framework (metal-candle uses Candle)

Think: metal-candle is to Candle as PyTorch Lightning is to PyTorch.

### metal-candle vs llama.cpp

- **metal-candle**: LoRA training + inference, Rust ecosystem
- **llama.cpp**: Inference-focused, quantization, C++ ecosystem

Use metal-candle for training, llama.cpp for highly optimized inference.

### metal-candle vs MLX

- **metal-candle**: Pure Rust, type-safe, single binary deployment
- **MLX**: Python, broader ecosystem, highly optimized kernels

Use metal-candle for Rust projects requiring type safety and simple deployment. Use MLX for maximum raw performance in Python environments.

## Still have questions?

- Check [Troubleshooting](./troubleshooting.md)
- Browse [User Guide](../guide/devices.md)
- Open an [issue](https://github.com/GarthDB/metal-candle/issues)
