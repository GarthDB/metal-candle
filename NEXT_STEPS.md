# metal-candle Next Steps

**Current Version**: v1.3.0  
**Date**: December 18, 2024  
**Status**: 🎉 PR #48 Merged, v1.3.0 Released

This document provides immediate next steps for metal-candle development.

---

## ✅ Recently Completed (v1.3.0)

- ✅ Enhanced streaming inference with `StreamToken` metadata
- ✅ Async streaming API (feature-gated)
- ✅ LoRA Adapter Registry for managing multiple adapters
- ✅ Complete documentation and examples
- ✅ 195+ tests passing
- ✅ Performance benchmarks validated
- ✅ PR #48 merged and tagged

---

## 🚀 Immediate Actions (This Week)

### 1. Post-Release Tasks

#### a. Create GitHub Release
```bash
gh release create v1.3.0 \
  --title "v1.3.0 - Streaming Inference and LoRA Adapter Management" \
  --notes-file release_notes_v1.3.0.md \
  --latest
```

**Checklist**:
- [ ] Update CHANGELOG date from `2025-01-XX` to actual date
- [ ] Create release notes highlighting:
  - Streaming inference features
  - Adapter management capabilities
  - Performance metrics
  - Breaking changes and migration guide

#### b. Publish to crates.io (Optional)

**Pre-publish checklist**:
- [ ] Verify all examples run correctly
- [ ] Update `Cargo.toml` metadata (description, keywords, categories)
- [ ] Ensure LICENSE file is current
- [ ] Run `cargo publish --dry-run`
- [ ] If dry-run succeeds: `cargo publish`

**Note**: Can delay crates.io publication until v1.3.1 for more stability

#### c. Social Media / Announcement

- [ ] Create announcement post highlighting:
  - Real-time streaming with metadata
  - Efficient adapter management
  - Performance improvements
  - Call for contributions

---

## 📋 v1.3.1 Planning (Next 2-4 Weeks)

### Priority 1: ApplyAdapter Implementation (#49)

**Issue**: https://github.com/GarthDB/metal-candle/issues/49  
**Estimate**: 1-2 weeks  
**Status**: 🔴 Critical - Completes v1.3.0 feature

#### Implementation Steps

1. **Week 1: Core Implementation**
   - [ ] Add adapter state tracking to `Qwen2ForSequenceClassification`
   - [ ] Implement `ApplyAdapter` trait methods
   - [ ] Create adapter application mechanism for attention layers
   - [ ] Create adapter application mechanism for MLP layers
   - [ ] Add unit tests for each component

2. **Week 2: Integration & Testing**
   - [ ] End-to-end adapter swapping tests
   - [ ] Memory leak detection tests
   - [ ] Performance benchmarks (switching time)
   - [ ] Update `adapter_swap_demo.rs` to use new API
   - [ ] Complete documentation

#### Success Criteria
- ✅ Models support `apply_adapter()` and `remove_adapter()`
- ✅ Adapter swapping works without model reload (<5ms)
- ✅ Zero memory leaks (validated with instruments)
- ✅ Test coverage ≥95% for new code
- ✅ Example demonstrates full workflow

#### Files to Modify
```
src/models/qwen.rs                  # Add ApplyAdapter impl
src/training/apply_adapter.rs       # Update trait docs
examples/adapter_swap_demo.rs       # Update to use ApplyAdapter
tests/training/adapter_hotswap.rs   # Add integration tests
CHANGELOG.md                        # Document v1.3.1 changes
```

### Priority 2: Streaming Performance Benchmarks

**Estimate**: 3-5 days  
**Status**: 🟡 High - Validates v1.3.0 claims

#### Tasks
- [ ] Add `benches/streaming.rs` benchmark suite
- [ ] Measure callback overhead per token
- [ ] Compare sync vs async streaming performance
- [ ] Test various buffer sizes
- [ ] Profile with Instruments.app
- [ ] Document results in `CHANGELOG.md`

#### Success Criteria
- ✅ Streaming overhead <5% (validate claim)
- ✅ Async streaming within 10% of sync
- ✅ Benchmark integrated into CI
- ✅ Results reproducible

#### Expected Results
```
Streaming Overhead Benchmarks:
- generate():           1000 tokens in 2.50s (400 tok/s) [baseline]
- generate_stream():    1000 tokens in 2.60s (385 tok/s) [+4% overhead] ✅
- generate_stream_async(): 1000 tokens in 2.65s (377 tok/s) [+6% overhead] ✅
```

### Priority 3: Performance Optimizations (If Needed)

**Estimate**: 2-3 days (conditional)  
**Status**: 🟢 Medium - Only if benchmarks reveal issues

#### Potential Areas
- [ ] Reduce allocations in streaming callback path
- [ ] Optimize `StreamToken` creation
- [ ] Cache tokenizer decoding
- [ ] Profile adapter switching hot paths

---

## 🎯 v1.4.0 Planning (February 2025)

### GGUF Format Support (#38)

**Priority**: 🔴 Critical - Most requested feature  
**Estimate**: 3-4 weeks

#### Research Phase (Week 1)
- [ ] Evaluate GGUF parsing libraries (`gguf-rs`, `llama-cpp-rs`)
- [ ] Study GGUF format specification
- [ ] Test loading reference GGUF models
- [ ] Design integration with existing loader

#### Implementation Phase (Week 2-3)
- [ ] Implement `GGUFLoader` struct
- [ ] Add quantization format support (Q4_0, Q8_0)
- [ ] Create dequantization logic
- [ ] Auto-detect GGUF vs safetensors
- [ ] Test with popular models (LLaMA, Mistral, Qwen GGUF)

#### Testing & Documentation (Week 4)
- [ ] Comprehensive test suite
- [ ] Performance benchmarks vs safetensors
- [ ] Memory usage validation
- [ ] Complete API documentation
- [ ] Update examples

#### Success Criteria
- ✅ Load GGUF models from HuggingFace
- ✅ Quantized inference works correctly
- ✅ Memory usage matches expectations (4x reduction for Q4)
- ✅ Performance within 10% of safetensors fp16
- ✅ Examples demonstrate GGUF loading

---

## 📊 Metrics to Track

### Code Quality
- **Tests**: 195+ → target 220+ by v1.3.1
- **Coverage**: 80%+ maintained
- **Clippy Warnings**: 0 (strict)
- **Documentation**: 100% for public APIs

### Performance (M4 Max Baseline)
- **Adapter Loading**: 2.5ms (rank 8) ✅
- **Adapter Switching**: 1.3ms ✅
- **Streaming Overhead**: <5% (to validate)
- **LoRA Forward Pass**: No regression ✅

### Community
- **GitHub Stars**: Track growth
- **Issues**: Response time <48h target
- **PRs**: Review time <1 week target
- **Downloads**: Monitor crates.io (after publication)

---

## 🤝 Contribution Opportunities

### Good First Issues
- [ ] Add more model architecture examples
- [ ] Improve error messages with suggestions
- [ ] Add visualization tools for adapter weights
- [ ] Create Jupyter notebook examples
- [ ] Improve documentation with diagrams

### Advanced Contributions
- [ ] GGUF loader implementation
- [ ] Flash Attention Metal kernel
- [ ] Additional model architectures (LLaMA, Mistral)
- [ ] Quantization methods (GPTQ, AWQ)
- [ ] Multi-GPU support research

---

## 📅 Release Timeline

| Version | Focus | Target Date | Status |
|---------|-------|-------------|--------|
| v1.3.0 | Streaming & Adapters | Dec 18, 2024 | ✅ Released |
| v1.3.1 | ApplyAdapter | Late Jan 2025 | 🚧 In Progress |
| v1.4.0 | GGUF Support | Late Feb 2025 | 📋 Planned |
| v1.5.0 | Multi-Arch | Late Mar 2025 | 📋 Planned |
| v1.6.0 | Quantization | Late Apr 2025 | 📋 Planned |
| v1.7.0 | Flash Attention | Late May 2025 | 📋 Planned |
| v2.0.0 | Multi-GPU | Q3 2025 | 📋 Planned |

---

## 🔗 Quick Links

- **GitHub Repository**: https://github.com/GarthDB/metal-candle
- **Project Board**: https://github.com/users/GarthDB/projects/3
- **Issue #49 (ApplyAdapter)**: https://github.com/GarthDB/metal-candle/issues/49
- **Full Roadmap**: [ROADMAP.md](ROADMAP.md)
- **CHANGELOG**: [CHANGELOG.md](CHANGELOG.md)

---

## 💡 Questions?

- Open an issue: https://github.com/GarthDB/metal-candle/issues/new
- Start a discussion: https://github.com/GarthDB/metal-candle/discussions
- Review CONTRIBUTING.md for development guidelines

---

**Next Update**: After v1.3.1 release (late January 2025)

