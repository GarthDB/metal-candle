# metal-candle Release Roadmap

**Last Updated**: December 18, 2024  
**Current Version**: v1.3.0  
**Status**: ✅ Production Ready

This document outlines the planned releases for metal-candle. For detailed task tracking, see the [GitHub Project Board](https://github.com/users/GarthDB/projects/3).

---

## Release Strategy

### Version Numbering (SemVer)

- **Major (X.0.0)**: Breaking API changes, architectural shifts
- **Minor (1.X.0)**: New features, backward-compatible additions
- **Patch (1.3.X)**: Bug fixes, performance improvements, docs

### Release Cadence

- **Minor releases**: Every 2-4 weeks (feature-driven)
- **Patch releases**: As needed (bug fixes)
- **Major releases**: When breaking changes necessary (rare)

### Quality Gates (All Releases)

- ✅ All tests passing (100%)
- ✅ Code coverage ≥80%
- ✅ Zero clippy pedantic warnings
- ✅ Complete API documentation
- ✅ Updated CHANGELOG.md
- ✅ Migration guide for breaking changes

---

## v1.3.1 - LoRA Hot-Swapping (January 2025)

**Focus**: Complete `ApplyAdapter` implementation and streaming benchmarks

### Objectives

1. Full hot-swapping of LoRA adapters on models
2. Comprehensive streaming performance benchmarks
3. Performance optimizations based on benchmark findings

### Features

#### 1. ApplyAdapter Implementation (#49)

**Priority**: 🔴 Critical (completes v1.3.0 feature)

**Scope**:
- Implement `ApplyAdapter` trait for `Qwen2ForSequenceClassification`
- Add adapter state tracking to models
- Create adapter application mechanism for attention/MLP layers
- Support multiple concurrent adapters per model
- Add comprehensive tests for adapter swapping

**Success Criteria**:
- ✅ Models support `apply_adapter()` and `remove_adapter()`
- ✅ Adapter swapping works without model reload
- ✅ Zero memory leaks during adapter switching
- ✅ `adapter_swap_demo.rs` fully functional
- ✅ Test coverage ≥95% for new code

**Breaking Changes**: None (additive feature)

#### 2. Streaming Benchmarks (#TBD)

**Priority**: 🟡 High (validate v1.3.0 performance claims)

**Scope**:
- Add streaming overhead benchmarks to `benches/inference.rs`
- Measure callback overhead per token
- Compare sync vs async streaming performance
- Benchmark streaming with various buffer sizes
- Document results in CHANGELOG

**Success Criteria**:
- ✅ Streaming overhead <5% (as claimed)
- ✅ Async streaming performance within 10% of sync
- ✅ Benchmarks integrated into CI
- ✅ Results documented and reproducible

#### 3. Performance Optimizations

**Priority**: 🟢 Medium (based on benchmark findings)

**Potential Areas**:
- Reduce allocations in streaming path
- Optimize `StreamToken` creation
- Improve adapter switching efficiency
- Profile and optimize hot paths

**Success Criteria**:
- ✅ No regressions vs v1.3.0
- ✅ Measurable improvements in identified bottlenecks

### Timeline

- **Week 1-2**: ApplyAdapter implementation
- **Week 3**: Streaming benchmarks
- **Week 4**: Optimizations and polish
- **Target Release**: Mid-Late January 2025

### Documentation

- Complete `ApplyAdapter` API docs
- Update adapter management guide
- Add streaming performance section to BENCHMARKS.md
- Create migration examples

---

## v1.4.0 - GGUF Support (February 2025)

**Focus**: Load quantized GGUF models for inference

### Objectives

1. Support GGUF format loading (llama.cpp ecosystem)
2. Enable quantized inference (4-bit, 8-bit)
3. Maintain performance parity with safetensors

### Features

#### 1. GGUF Format Support (#38)

**Priority**: 🔴 Critical (most requested feature)

**Scope**:
- Implement `GGUFLoader` for model loading
- Support GGUF metadata parsing
- Handle quantized weight formats (Q4_0, Q8_0)
- Auto-detect GGUF vs safetensors based on file extension
- Convert GGUF weights to Candle tensors

**Success Criteria**:
- ✅ Load GGUF models (LLaMA, Mistral, Qwen)
- ✅ Quantized inference works correctly
- ✅ Memory usage matches expectations
- ✅ Performance within 10% of safetensors
- ✅ Examples demonstrate GGUF loading

**Technical Notes**:
- Use `gguf-rs` crate for parsing
- Implement dequantization on-the-fly or in Metal
- Support lazy loading for large models

#### 2. Quantized Inference

**Priority**: 🟡 High (enables smaller memory footprint)

**Scope**:
- Support 4-bit and 8-bit quantized models
- Implement dequantization kernels
- Measure performance vs fp16
- Document memory savings

**Success Criteria**:
- ✅ 4-bit models use ~4x less memory
- ✅ Inference speed within 20% of fp16
- ✅ Quality/perplexity within acceptable range

#### 3. Model Compatibility Testing

**Priority**: 🟡 High

**Scope**:
- Test popular GGUF models (LLaMA 2/3, Mistral, Qwen)
- Validate outputs against reference implementations
- Document supported architectures

### Breaking Changes

None (additive feature)

### Timeline

- **Week 1-2**: GGUF parsing and loading
- **Week 3**: Quantization support
- **Week 4**: Testing and documentation
- **Target Release**: Late February 2025

### Dependencies

- `gguf-rs` or custom GGUF parser
- Updated Candle version (if needed)

---

## v1.5.0 - Multi-Architecture Support (March 2025)

**Focus**: Support LLaMA, Mistral, and other popular architectures

### Objectives

1. Expand beyond Qwen to popular open models
2. Create generic transformer architecture abstraction
3. Enable LoRA training for all supported architectures

### Features

#### 1. LLaMA Architecture (#39)

**Priority**: 🔴 Critical (most popular architecture)

**Scope**:
- Implement LLaMA 2/3 model architecture
- Support RoPE embeddings
- Add GQA (Grouped Query Attention)
- Enable LoRA training for LLaMA

**Success Criteria**:
- ✅ Load LLaMA 2/3 models (safetensors and GGUF)
- ✅ Inference matches reference implementation
- ✅ LoRA training works end-to-end
- ✅ Performance competitive with llama.cpp

#### 2. Mistral Architecture (#39)

**Priority**: 🟡 High

**Scope**:
- Implement Mistral model architecture
- Support sliding window attention
- Enable LoRA training for Mistral

**Success Criteria**:
- ✅ Load Mistral models
- ✅ Correct inference output
- ✅ LoRA training functional

#### 3. Generic Transformer Abstraction

**Priority**: 🟢 Medium (enables future architectures)

**Scope**:
- Extract common transformer components
- Create trait-based architecture system
- Reduce code duplication across models

**Success Criteria**:
- ✅ All models use shared components where possible
- ✅ Easy to add new architectures
- ✅ No performance regressions

### Breaking Changes

**Potential**: Model loading API may change to accommodate multiple architectures

**Mitigation**: Provide compatibility layer and migration guide

### Timeline

- **Week 1-2**: LLaMA implementation
- **Week 3**: Mistral implementation
- **Week 4**: Abstraction and testing
- **Target Release**: Late March 2025

---

## v1.6.0 - Advanced Quantization (April 2025)

**Focus**: In-memory quantization and performance optimization

### Objectives

1. Support quantizing models in-memory (not just loading)
2. Implement advanced quantization methods (GPTQ, AWQ)
3. Optimize quantized inference performance

### Features

#### 1. In-Memory Quantization (#40)

**Priority**: 🟡 High

**Scope**:
- Quantize fp16 models to 4-bit/8-bit at runtime
- Support post-training quantization (PTQ)
- Implement calibration for accuracy

**Success Criteria**:
- ✅ Convert models to quantized formats
- ✅ Save quantized models
- ✅ Accuracy loss <2% on benchmarks

#### 2. Advanced Quantization Methods (#40)

**Priority**: 🟢 Medium

**Scope**:
- GPTQ (layer-wise quantization)
- AWQ (activation-aware weight quantization)
- Compare methods for quality/speed

**Success Criteria**:
- ✅ Multiple quantization methods available
- ✅ Documentation on method selection
- ✅ Benchmarked trade-offs

#### 3. Quantized Metal Kernels

**Priority**: 🔴 Critical (performance)

**Scope**:
- Custom Metal kernels for quantized operations
- Optimize int4/int8 matmul
- Reduce dequantization overhead

**Success Criteria**:
- ✅ Quantized inference within 10% of fp16 speed
- ✅ Memory usage matches quantization level
- ✅ Profiled and optimized

### Timeline

- **Week 1-2**: PTQ implementation
- **Week 3**: Advanced methods (GPTQ/AWQ)
- **Week 4**: Metal kernel optimization
- **Target Release**: Late April 2025

---

## v1.7.0 - Flash Attention (May 2025)

**Focus**: 2-4x faster attention for long contexts

### Objectives

1. Implement Flash Attention algorithm
2. Support long-context inference (32k+ tokens)
3. Maintain numerical stability

### Features

#### 1. Flash Attention Implementation (#41)

**Priority**: 🔴 Critical (major performance win)

**Scope**:
- Port Flash Attention 2 algorithm to Metal
- Support causal and bidirectional attention
- Integrate with existing models

**Success Criteria**:
- ✅ 2-4x speedup for attention
- ✅ Supports sequence lengths up to 32k
- ✅ Numerically stable
- ✅ Memory usage reduced vs standard attention

#### 2. Long-Context Optimizations

**Priority**: 🟡 High

**Scope**:
- Optimize KV-cache for long sequences
- Implement efficient attention masking
- Support sliding window attention

**Success Criteria**:
- ✅ Efficient inference on 16k+ token contexts
- ✅ Memory usage scales linearly
- ✅ Performance competitive with leading implementations

### Timeline

- **Week 1-3**: Flash Attention Metal kernel
- **Week 4**: Integration and testing
- **Target Release**: Late May 2025

---

## v2.0.0 - Multi-GPU & Advanced Features (Q3 2025)

**Focus**: Scale to 70B+ models, production-grade deployment

### Objectives

1. Multi-GPU training and inference
2. Model parallelism strategies
3. Production deployment features

### Features

#### 1. Multi-GPU Support (#42)

**Priority**: 🔴 Critical (enables large models)

**Scope**:
- Data parallelism for training
- Tensor parallelism for inference
- Pipeline parallelism for very large models
- Efficient gradient synchronization

**Success Criteria**:
- ✅ Train 70B+ models on multiple GPUs
- ✅ Linear scaling up to 4 GPUs
- ✅ Efficient memory usage
- ✅ Works with Ultra interconnect

#### 2. Model Parallelism

**Priority**: 🟡 High

**Scope**:
- Split model across GPUs
- Automatic sharding strategies
- Efficient cross-GPU communication

#### 3. Production Features

**Priority**: 🟢 Medium

**Scope**:
- Model serving optimizations
- Batched inference with dynamic batching
- Request queuing and scheduling
- Monitoring and metrics

### Breaking Changes

**Major**: Multi-GPU API differs significantly from single-GPU

**Migration**: Comprehensive guide and compatibility layer

### Timeline

- **Q3 2025**: 8-12 weeks of development
- Multiple betas before stable release

---

## Future Considerations (v2.1+)

### Potential Features

- **Model Distillation**: Compress large models to smaller ones
- **Speculative Decoding**: Faster inference via draft models
- **MoE (Mixture of Experts)**: Mixtral-style architectures
- **Vision Models**: Support for multimodal models
- **Fine-Tuning Methods**: QLoRA, IA³, prefix tuning
- **Training Optimizations**: Gradient checkpointing, mixed precision
- **Deployment Tools**: Model serving, API wrappers
- **Cross-Platform**: Linux/Windows Metal alternatives (Vulkan?)

### Community-Driven

Future features will be prioritized based on:
1. Community votes (👍 on GitHub issues)
2. Production use cases
3. Upstream Candle improvements
4. Apple Silicon hardware capabilities

---

## GitHub Project Board

Track progress and vote on features: [v1.3+ Roadmap](https://github.com/users/GarthDB/projects/3)

### Open Issues

- #38 - GGUF format support
- #39 - LLaMA/Mistral architectures
- #40 - 4-bit/8-bit quantization
- #41 - Flash Attention
- #42 - Multi-GPU support

### How to Influence Priorities

1. 👍 Vote on issues you care about
2. Comment with your use case
3. Contribute PRs for features
4. Sponsor development of specific features

---

## Completion Status

### ✅ Completed

- **v1.0.0** (Dec 10, 2024): Initial release - LoRA training, Qwen models, Metal backend
- **v1.1.0** (Dec 11, 2024): Custom operations, lazy execution, performance improvements
- **v1.2.0** (Dec 11, 2024): Embeddings support (BERT), HuggingFace Hub integration
- **v1.2.1** (Dec 11, 2024): Benchmark configuration fixes, CI improvements
- **v1.3.0** (Jan 2025): Streaming inference, LoRA adapter management

### 🚧 In Progress

- **v1.3.1** (Jan 2025): ApplyAdapter implementation, streaming benchmarks

### 📋 Planned

- **v1.4.0** (Feb 2025): GGUF support
- **v1.5.0** (Mar 2025): Multi-architecture support
- **v1.6.0** (Apr 2025): Advanced quantization
- **v1.7.0** (May 2025): Flash Attention
- **v2.0.0** (Q3 2025): Multi-GPU support

---

## Contributing

Want to help build these features? See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development standards
- Testing requirements
- PR process
- Code quality gates

All contributions welcome! 🚀

---

**Maintained by**: [@GarthDB](https://github.com/GarthDB)  
**License**: Apache-2.0  
**Project**: [metal-candle on GitHub](https://github.com/GarthDB/metal-candle)

