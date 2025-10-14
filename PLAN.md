# metal-candle: Pure Rust ML for Apple Silicon

## Overview

`metal-candle` is a production-quality, general-purpose Rust crate for machine learning on Apple Silicon that replaces MLX+PyO3. Built on Candle with Metal backend, featuring LoRA training, multiple model format support, and production-grade quality standards.

**Repository**: Standalone crate (to be published to crates.io)  
**Target**: Replace Ferris project's PyO3+MLX Python dependency with pure Rust  
**Timeline**: 12 weeks to v1.0

## Motivation

- **Deployment Simplicity**: Single binary distribution (no Python runtime)
- **Performance**: Eliminate PyO3 overhead, native Rust-to-Metal calls
- **Development Experience**: Stay in Rust ecosystem with better tooling
- **Long-term Support**: Multiple model formats, extensible architecture

## Scope

### Core Features for v1.0

- LoRA training pipeline for transformer models
- Model loading (safetensors primary, extensible to GGUF/others)
- Metal-accelerated inference and training
- Qwen2.5-Coder model family support
- Checkpoint management and serialization

### Quality Standards

- **Clippy**: Pedantic level (not nursery) - zero warnings enforced
- **Code Coverage**: ≥80% line coverage with comprehensive tests
- **Performance**: Within 90-100% of MLX+PyO3 baseline (benchmarked locally on real hardware)
- **Documentation**: Complete API docs, examples, architecture guide

### Out of Scope for v1.0

- Multi-GPU/distributed training
- Non-transformer architectures
- Quantization below fp16 (can add in v1.1+)
- Windows/Linux support (Apple Silicon focus)

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                    metal-candle                         │
├─────────────────────────────────────────────────────────┤
│  Public API Layer                                       │
│  ├── ModelLoader (multi-format)                         │
│  ├── LoRATrainer (training pipeline)                    │
│  ├── Generator (inference)                              │
│  └── Checkpoint (save/load)                             │
├─────────────────────────────────────────────────────────┤
│  Model Layer                                            │
│  ├── Qwen2.5-Coder                                      │
│  ├── Generic Transformer                                │
│  └── LoRA Adapters                                      │
├─────────────────────────────────────────────────────────┤
│  Backend Layer (Candle)                                 │
│  ├── Metal Device                                       │
│  ├── Tensor Operations                                  │
│  └── Optimizers                                         │
├─────────────────────────────────────────────────────────┤
│  Apple Metal Performance Shaders                        │
└─────────────────────────────────────────────────────────┘
```

### Crate Structure

```
metal-candle/
├── Cargo.toml                  # Standalone crate, strict lints
├── PLAN.md                     # This file
├── README.md                   # Usage guide
├── ARCHITECTURE.md             # Design decisions
├── .github/
│   └── workflows/
│       └── ci.yml              # Apple Silicon only (macos-14)
├── src/
│   ├── lib.rs                  # Public API exports
│   ├── error.rs                # Error types
│   ├── backend/
│   │   ├── mod.rs
│   │   ├── metal.rs            # Metal device abstraction
│   │   └── tensor.rs           # Candle tensor wrappers
│   ├── models/
│   │   ├── mod.rs
│   │   ├── loader.rs           # Multi-format model loading
│   │   ├── qwen.rs             # Qwen architecture
│   │   └── transformer.rs      # Generic transformer components
│   ├── training/
│   │   ├── mod.rs
│   │   ├── lora.rs             # LoRA adapter implementation
│   │   ├── optimizer.rs        # Optimizers (AdamW, etc.)
│   │   └── trainer.rs          # Training loop orchestration
│   ├── inference/
│   │   ├── mod.rs
│   │   └── generator.rs        # Text generation pipeline
│   └── checkpoint/
│       ├── mod.rs
│       ├── safetensors.rs      # Safetensors I/O
│       └── manager.rs          # Checkpoint versioning
├── benches/
│   ├── mlx_comparison.rs       # Local benchmark vs MLX+PyO3
│   └── training.rs             # Training performance
├── tests/
│   ├── integration/
│   │   ├── training.rs
│   │   └── inference.rs
│   └── models/
│       └── loading.rs
└── examples/
    ├── train_lora.rs           # Complete LoRA training example
    ├── inference.rs            # Inference example
    └── load_model.rs           # Model loading example
```

### Key Dependencies

```toml
[dependencies]
# Candle - ML framework with Metal backend
candle-core = { version = "0.7", features = ["metal"] }
candle-nn = "0.7"

# Model formats
safetensors = "0.4"
tokenizers = "0.20"  # HuggingFace tokenizers

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"

[dev-dependencies]
criterion = "0.5"  # Local benchmarking
tempfile = "3.0"
approx = "0.5"     # Float comparison in tests

[build-dependencies]
# None initially
```

## Implementation Phases

### Phase 1: Foundation & Metal Backend (Week 1-2)

**Objectives:**
- Set up standalone crate with strict quality gates
- Implement Metal device abstraction using Candle
- Create tensor operations wrapper for common ML ops
- Establish CI/CD with clippy pedantic and coverage enforcement

**Key Deliverables:**

1. **Project Setup**
   - `Cargo.toml` with strict lints (clippy pedantic, deny warnings)
   - CI workflow targeting Apple Silicon (macos-14 GitHub runner)
   - Basic README with project vision

2. **Metal Backend** (`src/backend/`)
   - Device initialization and detection
   - Memory management utilities
   - Tensor operation wrappers (matmul, softmax, layer_norm, etc.)
   - Unit tests for each operation

3. **CI Pipeline** (`.github/workflows/ci.yml`)
   - Clippy pedantic (zero warnings)
   - Code coverage check (≥80% enforced)
   - Unit tests on Apple Silicon
   - **Note**: Performance benchmarks run locally only (see Benchmarking section)

**Success Criteria:**
- Metal device detection working on Apple Silicon
- Basic tensor operations validated against known results
- CI pipeline enforcing quality standards (clippy + coverage)
- Zero clippy pedantic warnings

**Files Created:**
```
src/lib.rs
src/error.rs
src/backend/mod.rs
src/backend/metal.rs
src/backend/tensor.rs
tests/backend_tests.rs
.github/workflows/ci.yml
```

### Phase 2: Model Loading & Architecture (Week 3-4)

**Objectives:**
- Implement safetensors model loading with validation
- Build Qwen2.5-Coder architecture using Candle primitives
- Create reusable transformer components
- Support model configuration from JSON

**Key Deliverables:**

1. **Model Loader** (`src/models/loader.rs`)
   - Safetensors format reader
   - Weight validation and shape checking
   - Config.json parsing
   - Error handling for corrupted/incompatible models

2. **Transformer Components** (`src/models/transformer.rs`)
   - Multi-head attention with rotary embeddings
   - MLP (feed-forward) layers
   - Layer normalization
   - Embedding layers
   - Generic and reusable

3. **Qwen Architecture** (`src/models/qwen.rs`)
   - Qwen2.5-Coder specific configuration
   - Model forward pass
   - Attention mask handling
   - Tested against reference outputs

**Success Criteria:**
- Load existing Qwen2.5-Coder checkpoints from Ferris data
- Forward pass produces outputs matching MLX implementation (within fp16 tolerance)
- Memory usage ≤ current MLX implementation
- Full documentation for all public APIs

**Files Created:**
```
src/models/mod.rs
src/models/loader.rs
src/models/transformer.rs
src/models/qwen.rs
tests/models/loading.rs
tests/models/forward_pass.rs
examples/load_model.rs
```

### Phase 3: LoRA Training Pipeline (Week 5-7)

**Objectives:**
- Implement LoRA adapters (low-rank adaptation)
- Build training loop with gradient computation
- Add AdamW optimizer with learning rate scheduling
- Create checkpoint saving/loading for LoRA weights

**Key Deliverables:**

1. **LoRA Adapters** (`src/training/lora.rs`)
   - LoRA layer implementation (A and B matrices)
   - Proper weight initialization (Gaussian for A, zeros for B)
   - Scaling factor (alpha/rank)
   - Gradient flow verification
   - Apply adapters to attention/MLP layers

2. **Training Loop** (`src/training/trainer.rs`)
   - Forward/backward pass orchestration
   - Loss computation (cross-entropy)
   - Gradient accumulation
   - Progress callbacks
   - Early stopping support
   - Memory-efficient batch processing

3. **Optimizer** (`src/training/optimizer.rs`)
   - AdamW implementation
   - Learning rate scheduling (warmup + cosine decay)
   - Gradient clipping
   - Weight decay

4. **Checkpoint Manager** (`src/checkpoint/manager.rs`)
   - Save LoRA weights (safetensors format)
   - Resume training from checkpoint
   - Metadata (epoch, step, loss, config)

**Success Criteria:**
- LoRA adapters trainable with verified gradient flow
- Training loop matches MLX behavior (loss convergence)
- Checkpoints saveable and resumable
- Memory efficient (monitor peak usage)
- Training example works end-to-end

**Files Created:**
```
src/training/mod.rs
src/training/lora.rs
src/training/trainer.rs
src/training/optimizer.rs
src/checkpoint/mod.rs
src/checkpoint/manager.rs
src/checkpoint/safetensors.rs
tests/training/lora_test.rs
tests/training/checkpoint_test.rs
examples/train_lora.rs
```

### Phase 4: Inference & Generation (Week 7-8)

**Objectives:**
- Implement text generation with multiple sampling strategies
- Add KV-cache for efficient inference
- Support batch inference
- Create ergonomic inference API

**Key Deliverables:**

1. **Text Generator** (`src/inference/generator.rs`)
   - Token-by-token generation
   - KV-cache implementation for efficiency
   - Sampling strategies:
     - Greedy
     - Top-k
     - Top-p (nucleus)
     - Temperature scaling
   - Batch inference support
   - Stopping criteria (max length, EOS token)

2. **Inference API**
   - Simple high-level API for common use cases
   - Streaming token generation (callback-based)
   - Configuration builder pattern

**Success Criteria:**
- Token generation speed ≥50 tokens/sec (comparable to MLX)
- KV-cache reduces redundant computation (measure speedup)
- API ergonomic and well-documented
- Inference example demonstrates common patterns

**Files Created:**
```
src/inference/mod.rs
src/inference/generator.rs
tests/inference/generation_test.rs
examples/inference.rs
```

### Phase 5: Quality, Benchmarking & Documentation (Week 9-10)

**Objectives:**
- Achieve ≥80% code coverage with comprehensive tests
- Run local performance benchmarks vs. MLX+PyO3 baseline
- Write complete API documentation and architecture guide
- Create examples for common workflows

**Key Deliverables:**

1. **Test Coverage**
   - Unit tests for all public APIs
   - Integration tests for end-to-end workflows
   - Edge case testing (OOM, invalid inputs, etc.)
   - Measured via `cargo llvm-cov` (≥80% enforced)

2. **Performance Benchmarks** (Local only - see Benchmarking Strategy)
   - Training throughput (tokens/sec)
   - Inference latency (ms/token, tokens/sec)
   - Memory usage (peak GB)
   - KV-cache speedup
   - Side-by-side comparison with MLX+PyO3
   - Results documented in `BENCHMARKS.md`

3. **Documentation**
   - Complete rustdoc for all public APIs
   - `README.md` with quickstart and examples
   - `ARCHITECTURE.md` with design decisions
   - `CONTRIBUTING.md` for future contributors
   - Example code for common patterns

**Success Criteria:**
- Code coverage ≥80% (measured, enforced)
- Performance within 90-100% of MLX+PyO3 (training and inference)
- Documentation complete and comprehensive
- Zero clippy pedantic warnings
- Ready for public release

**Files Created:**
```
ARCHITECTURE.md
BENCHMARKS.md
CONTRIBUTING.md
benches/mlx_comparison.rs
benches/training.rs
(comprehensive test coverage across all modules)
```

### Phase 6: Ferris Integration (Week 11-12)

**Objectives:**
- Publish `metal-candle` v1.0 to crates.io
- Replace ferris-mlx PyO3 implementation with metal-candle
- Migrate existing training data and checkpoints
- Update Ferris CLI and plugins to use new backend
- Verify all AI tools (generate-tests, generate-docs) working

**Key Deliverables:**

1. **Crate Publication**
   - Polish crate metadata (Cargo.toml)
   - Final documentation review
   - Publish to crates.io as v1.0.0

2. **Ferris Integration**
   - Replace `ferris-mlx` PyO3 implementation
   - Update `Cargo.toml` dependencies
   - Remove Python virtual environment requirements
   - Migrate model checkpoints (if format changed)
   - Update configuration files

3. **Validation**
   - All existing MLX functionality working
   - Performance meets or exceeds baseline
   - Single-binary deployment verified
   - AI tool integration tested (qwen plugin, etc.)

**Success Criteria:**
- `metal-candle` published and available on crates.io
- Ferris project builds without Python dependencies
- All tests passing in Ferris
- Performance validated on real workloads
- Deployment simplified (no Python runtime needed)

**Files Modified in Ferris:**
```
apps/ferris-mlx/Cargo.toml          # Replace pyo3 with metal-candle
apps/ferris-mlx/src/lib.rs          # Update to use metal-candle API
apps/ferris-mlx/src/bridge.rs       # Remove or simplify
plugins/core/qwen/                  # Update to metal-candle
docs/CANDLE_METAL_MIGRATION.md      # Migration guide
```

## Quality Gates

### Clippy Configuration

**Cargo.toml lints section:**
```toml
[lints.clippy]
# Strict pedantic linting (not nursery - too unstable)
pedantic = "deny"

# Cargo lints
cargo = "warn"

# Deny these specific lints
all = "deny"
correctness = "deny"
suspicious = "deny"
complexity = "deny"
perf = "deny"

# Allow these specific pedantic lints if needed
# (add as exceptions with justification)
# module_name_repetitions = "allow"  # Example
```

**CI Enforcement:**
- CI fails on any clippy warnings
- Local development uses `cargo clippy -- -D warnings`
- Regular clippy updates monitored

### Code Coverage

**Target: ≥80% line coverage**

**Tooling:**
```bash
# Generate coverage report
cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info

# View HTML report
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html
```

**CI Enforcement:**
- Coverage report generated on every PR
- PR fails if coverage drops below 80%
- Coverage badge in README.md

**Testing Strategy:**
- Unit tests: Each module's core functionality
- Integration tests: End-to-end workflows
- Property tests: Numerical stability, gradient checks
- Edge cases: OOM, invalid inputs, corrupted models

### Benchmarking Strategy

**Local Benchmarks Only** (GitHub Actions runners don't reflect real Metal performance)

**Setup:**
```bash
# Run benchmarks locally on Apple Silicon
cargo bench --bench mlx_comparison

# Compare against baseline
cargo bench --bench training -- --save-baseline main
```

**Benchmark Suites:**

1. **Training Benchmarks** (`benches/training.rs`)
   - LoRA training throughput (tokens/sec)
   - Memory usage during training
   - Gradient computation time
   - Checkpoint save/load time

2. **Inference Benchmarks** (`benches/inference.rs`)
   - Generation latency (ms/token)
   - Throughput (tokens/sec)
   - KV-cache speedup
   - Batch inference scaling

3. **MLX Comparison** (`benches/mlx_comparison.rs`)
   - Side-by-side with current MLX+PyO3 implementation
   - Same model, same data, same hardware
   - Document results in `BENCHMARKS.md`

**Performance Targets:**
- Training: Within 90-100% of MLX+PyO3 throughput
- Inference: Within 95-100% of MLX+PyO3 speed
- Memory: ≤ MLX+PyO3 peak usage
- Regression detection: Alert if >5% slower than baseline

**CI Approach:**
- CI runs lightweight sanity benchmarks (not for performance)
- Real benchmarks run locally and documented
- Manual performance testing before releases

## Model Format Support

### v1.0: Safetensors (Primary)

**Why Safetensors:**
- Modern, safe, well-supported by Candle
- Growing ecosystem standard
- Fast loading, memory-mapped
- Rust-native implementation available

**Implementation:**
- Use `safetensors` crate (0.4+)
- Validate tensor shapes and dtypes
- Support lazy loading for large models

### v1.1+: Extended Format Support

**Planned Additions:**
- **GGUF**: llama.cpp ecosystem compatibility
- **PyTorch bins**: Legacy support if needed
- **Custom format**: Optimized for deployment (quantized, fused)

**Design Pattern:**
```rust
pub trait ModelLoader {
    fn load(&self, path: &Path) -> Result<ModelWeights>;
    fn validate(&self, path: &Path) -> Result<ModelInfo>;
}

// Implementations:
// - SafetensorsLoader (v1.0)
// - GGUFLoader (v1.1)
// - PyTorchLoader (v1.1)
```

**Extensibility:**
- Abstract trait allows pluggable loaders
- No breaking changes to add new formats
- Auto-detection based on file extension/magic bytes

## Risk Mitigation

### Technical Risks

**1. Candle LoRA Maturity**
- **Risk**: Candle may not have built-in LoRA support
- **Mitigation**: Implement from scratch using PyTorch/MLX as reference
- **Fallback**: Use external LoRA crate if available and compatible

**2. Metal Performance Parity**
- **Risk**: Metal Candle backend may be slower than MLX
- **Mitigation**: Early benchmarking, profiling with Instruments
- **Fallback**: Optimize critical kernels, contribute to Candle upstream

**3. Model Checkpoint Compatibility**
- **Risk**: Existing checkpoints may not load correctly
- **Mitigation**: Extensive validation tests, conversion utilities
- **Fallback**: Provide checkpoint migration tool

**4. Memory Management**
- **Risk**: Memory leaks or excessive usage
- **Mitigation**: Regular profiling, memory tests
- **Fallback**: Implement gradient checkpointing, model sharding

### Timeline Risks

**1. LoRA Implementation Complexity**
- **Risk**: Custom LoRA implementation takes longer than estimated
- **Mitigation**: Prioritize minimal viable implementation
- **Fallback**: Extend Phase 3 by 1-2 weeks if needed

**2. Performance Optimization**
- **Risk**: Performance significantly below MLX+PyO3
- **Mitigation**: Profile early, identify bottlenecks
- **Fallback**: Ship v1.0 with known perf gap, optimize in v1.1

**3. Candle API Changes**
- **Risk**: Candle 0.7+ introduces breaking changes
- **Mitigation**: Pin to specific Candle version, monitor releases
- **Fallback**: Fork Candle if necessary (last resort)

### Mitigation Strategy Summary

- **Phased Delivery**: Each phase delivers working increment
- **Early Validation**: Benchmark and test continuously
- **Flexible Timeline**: Can extend critical phases if needed
- **Fallback Plans**: Identified for each major risk
- **Community Engagement**: Leverage Candle community for support

## Success Metrics

### Technical Metrics

**Quality:**
- ✅ Zero clippy pedantic warnings
- ✅ ≥80% code coverage
- ✅ All tests passing on Apple Silicon
- ✅ Documentation complete for public APIs

**Performance:**
- ✅ Training throughput: 90-100% of MLX+PyO3
- ✅ Inference speed: 95-100% of MLX+PyO3
- ✅ Memory usage: ≤ MLX+PyO3
- ✅ KV-cache provides ≥2x speedup for long sequences

**Functionality:**
- ✅ Successful Qwen2.5-Coder fine-tuning
- ✅ LoRA checkpoints saveable/resumable
- ✅ Model loading works for multiple formats (v1.1+)
- ✅ Inference generates coherent text

### Strategic Metrics

**Publication:**
- ✅ Published to crates.io as v1.0.0
- ✅ README with clear quickstart
- ✅ Examples demonstrating key features

**Ferris Integration:**
- ✅ Ferris builds without Python dependencies
- ✅ Single-binary deployment working
- ✅ All AI tools (generate-tests, etc.) functional
- ✅ Performance meets or exceeds PyO3+MLX baseline

**Community Impact:**
- 🎯 General-purpose crate useful beyond Ferris
- 🎯 Documentation encourages adoption
- 🎯 Foundation for future Rust ML on Apple Silicon

## Development Guidelines

### Code Style

**Follow Rust API Guidelines:**
- https://rust-lang.github.io/api-guidelines/

**Key Principles:**
- Prefer explicit over implicit
- Use builder patterns for complex configuration
- Provide sensible defaults
- Make common cases easy, complex cases possible
- Error messages should be actionable

**Example API Design:**
```rust
// Good: Builder pattern with defaults
let model = ModelLoader::new()
    .with_device(Device::metal(0)?)
    .with_dtype(DType::F16)
    .load("path/to/model")?;

// Good: Explicit, clear errors
pub enum LoadError {
    FileNotFound { path: PathBuf },
    InvalidFormat { reason: String },
    IncompatibleVersion { expected: String, found: String },
}
```

### Testing Strategy

**Test Pyramid:**
- Many unit tests (fast, isolated)
- Some integration tests (realistic workflows)
- Few benchmark tests (performance validation)

**Coverage Focus:**
- Public APIs: 100% coverage
- Core algorithms: 100% coverage
- Utilities: ≥80% coverage
- Examples: Manual validation

**Test Organization:**
```
tests/
├── integration/        # End-to-end workflows
│   ├── training.rs
│   └── inference.rs
├── models/             # Model-specific tests
│   └── loading.rs
└── backend/            # Backend tests
    └── metal.rs
```

### Documentation Standards

**Every public item must have:**
- Summary (one sentence)
- Description (what it does, when to use)
- Examples (simple, runnable)
- Errors section (if fallible)
- Panics section (if can panic)

**Example:**
```rust
/// Loads a model from the given path.
///
/// Supports multiple formats (safetensors, GGUF). Format is auto-detected
/// based on file extension.
///
/// # Examples
///
/// ```no_run
/// use candle_metal::ModelLoader;
///
/// let model = ModelLoader::new().load("qwen2.5-coder.safetensors")?;
/// ```
///
/// # Errors
///
/// Returns `LoadError::FileNotFound` if the path doesn't exist.
/// Returns `LoadError::InvalidFormat` if the file is corrupted.
pub fn load(&self, path: impl AsRef<Path>) -> Result<Model, LoadError> {
    // ...
}
```

## Next Steps

### Immediate Actions (Day 1)

1. ✅ Create `~/Projects/metal-candle` directory
2. ✅ Write PLAN.md (this file)
3. ⬜ Initialize Git repository
4. ⬜ Create initial `Cargo.toml` with dependencies and strict lints
5. ⬜ Set up GitHub repository (if hosting on GitHub)
6. ⬜ Create `.github/workflows/ci.yml` for Apple Silicon CI

### Phase 1 Kickoff (Day 2-3)

1. ⬜ Implement basic project structure (lib.rs, error.rs)
2. ⬜ Set up Metal backend (backend/metal.rs)
3. ⬜ Write first unit tests
4. ⬜ Verify CI pipeline working
5. ⬜ Document Metal device abstraction

### Weekly Milestones

- **Week 2**: Metal backend complete, CI operational
- **Week 4**: Model loading works, Qwen forward pass validated
- **Week 7**: LoRA training works end-to-end
- **Week 8**: Inference optimized with KV-cache
- **Week 10**: Quality gates met, benchmarks complete
- **Week 12**: Published to crates.io, integrated into Ferris

## References

### Documentation

- **Candle**: https://github.com/huggingface/candle
- **Safetensors**: https://github.com/huggingface/safetensors
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Qwen2.5**: https://github.com/QwenLM/Qwen2.5

### Ferris Project Context

- Current MLX implementation: `~/Projects/Ferris/apps/ferris-mlx/`
- MLX research: `~/Projects/Ferris/docs/MLX_INTEGRATION_RESEARCH.md`
- ADR: `~/Projects/Ferris/docs/ADR_MLX_INTEGRATION.md`

### Community

- Candle Discord: (find link)
- Rust ML Discord: (find link)

---

**Status**: Planning Complete ✅  
**Next**: Begin Phase 1 Implementation  
**Updated**: 2025-10-14

