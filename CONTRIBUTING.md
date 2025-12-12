# Contributing to metal-candle

Thank you for your interest in contributing to `metal-candle`! This document provides guidelines and standards for contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation Standards](#documentation-standards)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- **Rust** 1.75+ (stable toolchain)
- **Apple Silicon Mac** (M1/M2/M3/M4) for full testing
- **Git** for version control
- **GitHub CLI** (`gh`) recommended for PR management

### Quick Start

```bash
# Clone the repository
git clone https://github.com/GarthDB/metal-candle.git
cd metal-candle

# Run tests
cargo test

# Check code quality
cargo clippy -- -D warnings
cargo fmt --check

# Build documentation
cargo doc --open
```

## Development Setup

### Required Tools

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install clippy and rustfmt
rustup component add clippy rustfmt

# Install llvm-tools for coverage
rustup component add llvm-tools-preview

# Install cargo-llvm-cov for coverage reports
cargo install cargo-llvm-cov

# Install act for local CI testing (optional)
brew install act
```

### IDE Setup

**Recommended**: VSCode with rust-analyzer extension

`.vscode/settings.json`:
```json
{
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.extraArgs": ["--", "-D", "warnings"]
}
```

## Code Quality Standards

### Clippy - Zero Warnings Policy

All code must pass pedantic clippy with zero warnings:

```toml
[lints.clippy]
pedantic = "deny"
cargo = "warn"
all = "deny"
correctness = "deny"
suspicious = "deny"
complexity = "deny"
perf = "deny"
```

**Check before committing**:
```bash
cargo clippy -- -D warnings
```

**Allowed Exceptions** (must be justified):
```rust
// Document why the exception is needed
#[allow(clippy::cast_precision_loss)] // Parameter count is reasonable size
let ratio = total_params as f64 / frozen_params as f64;
```

### Code Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt

# Verify formatting
cargo fmt --check
```

### Error Handling

**Library Code**: Use `thiserror` for structured errors

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("operation failed: {reason}")]
    OperationFailed { reason: String },
    
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
```

**Examples/Tests**: `anyhow` is allowed

```rust
use anyhow::Result;

fn main() -> Result<()> {
    // Example code
    Ok(())
}
```

**Never use** (in library code):
- `.unwrap()` - Use `?` operator or explicit error handling
- `.expect()` - Only acceptable in tests with clear justification

### Rust Style Guidelines

1. **Explicit is better than implicit**
   ```rust
   // Good
   let device = Device::new_cpu();
   let tensor = Tensor::zeros((batch, seq), DType::F32, &device)?;
   
   // Bad
   let tensor = Tensor::zeros((batch, seq))?; // Implicit device
   ```

2. **Use meaningful names**
   ```rust
   // Good
   let attention_scores = query.matmul(&key.transpose())?;
   
   // Bad
   let x = q.matmul(&k.t())?;
   ```

3. **Prefer references over clones**
   ```rust
   // Good
   pub fn process(&self, input: &Tensor) -> Result<Tensor>
   
   // Bad
   pub fn process(&self, input: Tensor) -> Result<Tensor>
   ```

4. **Document numerical stability**
   ```rust
   /// Numerically stable softmax implementation.
   /// Subtracts max before exp to prevent overflow.
   pub fn softmax_stable(x: &Tensor) -> Result<Tensor> {
       let max = x.max_keepdim(-1)?;
       let exp = (x.broadcast_sub(&max))?.exp()?;
       // ...
   }
   ```

## Testing Requirements

### Test Coverage

**Targets**:
- Overall: ≥80%
- Public APIs: 100%
- Core algorithms: 100%
- Backend/utilities: ≥80%

**Measure coverage**:
```bash
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html
```

### Test Organization

```rust
// Unit tests (in same file as implementation)
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // Test code
    }
}
```

```rust
// Integration tests (in tests/ directory)
// tests/integration/my_feature.rs
use metal_candle::*;

#[test]
fn test_end_to_end_workflow() {
    // Integration test
}
```

### Test Patterns

**1. Unit Tests**:
```rust
#[test]
fn test_lora_initialization() {
    let lora = LoRALayer::new(512, 8).unwrap();
    assert_eq!(lora.rank(), 8);
}

#[test]
fn test_invalid_input_returns_error() {
    let result = LoRALayer::new(512, 0);
    assert!(matches!(result, Err(LoRAError::InvalidRank)));
}
```

**2. Property Tests** (future):
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_softmax_sums_to_one(values in prop::collection::vec(-10.0f32..10.0, 1..100)) {
        let result = softmax(&values);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
```

**3. Float Comparisons**:
```rust
// Use epsilon tolerance, not equality
assert!((actual - expected).abs() < 1e-7);

// NOT this:
assert_eq!(actual, expected); // Will trigger clippy::float_cmp
```

## Documentation Standards

### Public API Documentation

**Every public item must have**:
1. Summary (one sentence)
2. Description (what it does, when to use)
3. Examples (simple, runnable)
4. Errors section (if returns `Result`)
5. Panics section (if can panic)

**Template**:
```rust
/// Loads a model from the specified path.
///
/// Supports safetensors format with automatic format detection.
/// The model is loaded onto the specified device (Metal by default).
///
/// # Examples
///
/// ```no_run
/// use metal_candle::ModelLoader;
///
/// let model = ModelLoader::new()
///     .with_dtype(DType::F16)
///     .load("model.safetensors")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`ModelError::FileNotFound`] if the path doesn't exist.
/// Returns [`ModelError::InvalidFormat`] if the file is corrupted.
///
/// # Panics
///
/// This function does not panic.
pub fn load(&self, path: impl AsRef<Path>) -> Result<Model, ModelError> {
    // Implementation
}
```

### Module Documentation

```rust
//! Model loading and format handling.
//!
//! This module provides utilities for loading ML models from various formats.
//! The primary format is safetensors.
//!
//! # Examples
//!
//! ```no_run
//! use metal_candle::models::ModelLoader;
//!
//! let model = ModelLoader::new().load("model.safetensors")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
```

### Documentation Tests

All examples in documentation must compile and run:

```bash
cargo test --doc
```

## Pull Request Process

### Before Submitting

**Pre-commit checklist**:
- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test` passes (all tests)
- [ ] `cargo fmt` applied
- [ ] New code has tests
- [ ] Public APIs have documentation
- [ ] No `unwrap()` or `expect()` in library code

**Optional**: Test CI locally
```bash
act -j clippy  # Test clippy check
act -j test    # Test test suite
act -j fmt     # Test formatting
```

### PR Guidelines

1. **Create a feature branch**:
   ```bash
   git checkout -b phase-N-feature-name
   ```

2. **Write clear commit messages**:
   ```
   feat: add LoRA adapter implementation
   fix: correct attention mask shape in Qwen model
   docs: add examples for model loading
   test: add integration tests for training pipeline
   perf: optimize KV-cache memory usage
   ```

3. **Keep PRs focused**: One feature/fix per PR

4. **Update documentation**: If adding public APIs

5. **Add tests**: For new functionality

6. **Update CHANGELOG**: For user-facing changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No warnings from clippy
```

### Review Process

1. **Automated checks**: CI must pass (clippy, tests, format)
2. **Code review**: At least one approval required
3. **Documentation review**: Ensure completeness
4. **Testing review**: Verify coverage

### Merging

- **Squash merge** for feature branches
- **Rebase** if conflicts with main
- **Delete branch** after merge

## Benchmarking Guidelines

### When to Run Benchmarks

**CI Smoke Tests (Automatic)**:
- Run on every PR via GitHub Actions
- Low sample size (10 iterations)
- Detects major regressions (>20%)
- ⚠️ NOT for performance claims

**Official Benchmarks (Manual)**:
- Before every release
- Required for performance claims
- High sample size (100+ iterations)
- Multiple runs (5+), take median
- Documented environment

### Running CI Benchmarks

CI benchmarks run automatically on PRs but you can test locally:

```bash
# Run all benchmarks (low sample size, fast)
cargo bench --all-features -- --sample-size 10

# Run specific benchmark
cargo bench --bench fused_lora_bench -- --sample-size 10
```

**Important**: CI benchmarks have ±10-20% variance. They catch bugs, not subtle regressions.

### Running Official Benchmarks

For releases and performance validation:

```bash
# Full official benchmark suite (30-60 minutes)
./scripts/run_official_benchmarks.sh

# Quick test of benchmark script (2 runs, 10s cooldown)
./scripts/run_official_benchmarks.sh --quick

# Skip MLX comparison if not installed
./scripts/run_official_benchmarks.sh --no-mlx
```

**Requirements**:
1. Dedicated Apple Silicon hardware
2. System idle (<5% CPU usage)
3. Connected to power (not battery)
4. No other apps running
5. Cool system (wait between runs)

**See**: [docs/BENCHMARK_CI.md](docs/BENCHMARK_CI.md) for detailed methodology

### Benchmark Best Practices

1. **Document Environment**:
   ```rust
   // In benchmark comments
   // Hardware: M1 Max 32GB
   // OS: macOS 14.5
   // Date: 2024-12-10
   ```

2. **Use Warmup Iterations**:
   ```rust
   // Warmup (first runs may be slower)
   for _ in 0..10 {
       let _ = model.forward(&input);
   }
   
   // Actual benchmark
   let start = Instant::now();
   for _ in 0..100 {
       let _ = model.forward(&input)?;
   }
   let elapsed = start.elapsed();
   ```

3. **Report Variance**:
   ```rust
   // Good: Report multiple runs
   println!("Run 1: {:.2}µs", time1);
   println!("Run 2: {:.2}µs", time2);
   println!("Median: {:.2}µs", median);
   
   // Bad: Single run
   println!("Time: {:.2}µs", time);
   ```

4. **Performance Claims**:
   ```markdown
   ✅ Good: "3.2x faster on M1 Max (median of 5 runs, macOS 14.5)"
   ❌ Bad: "3.2x faster" (no context)
   ```

### Adding New Benchmarks

Create benchmark in `benches/`:

```rust
// benches/my_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use metal_candle::*;

fn benchmark_my_feature(c: &mut Criterion) {
    let device = Device::new_metal(0).expect("Metal required");
    
    c.bench_function("my_feature", |b| {
        b.iter(|| {
            // Benchmark code
            black_box(my_feature(&device))
        });
    });
}

criterion_group!(benches, benchmark_my_feature);
criterion_main!(benches);
```

Add to `Cargo.toml`:
```toml
[[bench]]
name = "my_benchmark"
harness = false
```

## Release Process

See [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md) for the complete release process, including:
- Pre-release validation
- Official benchmark execution
- Documentation updates
- Publishing to crates.io
- Post-release verification

### Quick Release Checklist

**Phase 1: Pre-Release** (1-2 days)
- [ ] All tests passing (clippy, tests, coverage ≥80%)
- [ ] Examples work
- [ ] Dependencies audited

**Phase 2: Benchmarks** (1 day)
- [ ] Environment prepared (idle system, power, cool)
- [ ] Official benchmarks run (5+ runs)
- [ ] Results analyzed (variance <10%)
- [ ] BENCHMARKS.md updated

**Phase 3: Documentation** (2-3 hours)
- [ ] CHANGELOG.md updated with version/date
- [ ] Version bumped in Cargo.toml
- [ ] Release notes drafted

**Phase 4: Release** (1 hour)
- [ ] Tag created and pushed
- [ ] Published to crates.io
- [ ] GitHub Release created

**Phase 5: Post-Release** (1 hour)
- [ ] Installation verified
- [ ] docs.rs build successful

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **Major** (2.0.0): Breaking API changes
- **Minor** (1.X.0): New features, backwards compatible
- **Patch** (1.1.X): Bug fixes only

## Project Structure

```
metal-candle/
├── .github/
│   └── workflows/        # CI/CD workflows
├── benches/              # Benchmark suites
├── docs/                 # mdBook documentation
├── examples/             # Example programs
├── src/
│   ├── backend/          # Device and tensor abstractions
│   ├── error.rs          # Error types
│   ├── inference/        # Generation infrastructure
│   ├── lib.rs            # Public API exports
│   ├── models/           # Model loading and architectures
│   └── training/         # LoRA training components
├── tests/                # Integration tests
├── ARCHITECTURE.md       # Architecture documentation
├── BENCHMARKS.md         # Performance benchmarks
├── CONTRIBUTING.md       # This file
├── Cargo.toml            # Package manifest
├── LICENSE               # Apache-2.0
├── PLAN.md               # Project roadmap
└── README.md             # Project overview
```

## Getting Help

- **Issues**: https://github.com/GarthDB/metal-candle/issues
- **Discussions**: https://github.com/GarthDB/metal-candle/discussions
- **Candle Discord**: For Candle framework questions

## Useful Commands

```bash
# Run all checks locally
cargo clippy -- -D warnings && cargo test && cargo fmt

# Test specific module
cargo test training

# Run benchmarks
cargo bench

# Build docs
cargo doc --open

# Check for outdated dependencies
cargo outdated

# Audit dependencies
cargo audit

# Profile with Instruments (macOS)
cargo instruments -t Allocations --example train_lora
```

## Common Pitfalls

### 1. Metal-Specific Issues

```rust
// BAD: F64 not supported on Metal
let tensor = Tensor::zeros((10, 10), DType::F64, &metal_device)?;

// GOOD: Use F32 or F16
let tensor = Tensor::zeros((10, 10), DType::F16, &metal_device)?;
```

### 2. Tensor Contiguity

```rust
// BAD: May cause "unexpected striding" errors on Metal
let transposed = tensor.transpose(0, 1)?;
let result = transposed.matmul(&other)?;

// GOOD: Ensure contiguous
let transposed = tensor.transpose(0, 1)?.contiguous()?;
let result = transposed.matmul(&other)?;
```

### 3. Dtype Conversions

```rust
// BAD: Implicit conversion might fail
let result = tensor_f32.add(&tensor_f16)?;

// GOOD: Explicit conversion
let tensor_f32_converted = tensor_f16.to_dtype(DType::F32)?;
let result = tensor_f32.add(&tensor_f32_converted)?;
```

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 license.

---

**Thank you for contributing to metal-candle!** 🎉

