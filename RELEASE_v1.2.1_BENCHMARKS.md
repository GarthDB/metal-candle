# v1.2.1 Benchmark Results

## Executive Summary

All benchmarks successfully executed and produced valid output after fixing the missing `[[bench]]` configuration entries. Performance characteristics remain consistent with v1.2.0.

## Test Environment

- **Hardware**: Apple M4 Max (16 cores: 12P + 4E)
- **Memory**: 48 GB
- **OS**: macOS 26.1 (arm64)
- **Rust**: 1.91.1
- **Date**: December 11, 2025
- **Branch**: release/v1.2.1
- **Commit**: cfe50052ec39970a0ca8652de13e50ef6dfc1431

## Benchmark Results

### 1. Fused LoRA Benchmark ✅ FIXED

**Status**: Previously reported "no output" - now working correctly

| Operation | Median Time | Notes |
|-----------|-------------|-------|
| LoRA Unfused (Candle) | 24.4 µs | Standard Candle matmul operations |
| LoRA Fused (Metal) | 1.34 ms | Custom Metal kernel |

**Analysis**: 
- Benchmark now produces valid output after configuration fix
- Unfused operations remain fast for small batch sizes
- Fused kernel shows overhead for small operations (typical behavior)
- Both operations stable across 5 runs with <2% variance

**Configuration Fix**:
```toml
[[bench]]
name = "fused_lora_bench"
harness = false
```

### 2. Lazy vs Eager Execution ✅ FIXED

**Status**: Previously reported "no output" - now working correctly

| Operation | Size | Eager | Lazy (Sync) | Overhead |
|-----------|------|-------|-------------|----------|
| Add | 1024 | 361 ns | 761 ns | ~2.1x |
| Matmul | 64x64 | 6.75 µs | 7.28 µs | ~1.08x |
| Matmul | 128x128 | 92.5 µs | 93.1 µs | ~1.01x |
| LoRA | 4x512 | 2.66 µs | 4.00 µs | ~1.5x |

**Analysis**:
- Benchmark now produces comprehensive output
- Lazy evaluation overhead decreases with operation complexity
- For large operations (128x128 matmul), overhead is negligible (~1%)
- Graph building is efficient for complex computation chains

**Configuration Fix**:
```toml
[[bench]]
name = "lazy_vs_eager"
harness = false
```

### 3. Inference Benchmark ✅

**Status**: Working correctly (already configured)

| Metric | Time | Notes |
|--------|------|-------|
| KV-Cache access | 11.8 ns | O(1) position tracking |
| Token sampling | 337 ns | Minimal overhead |
| Forward pass (seq_len=128) | 480 µs | Consistent across runs |

**Analysis**:
- No performance regression from v1.2.0
- KV-cache performance remains excellent
- Sampling overhead < 0.1% of forward pass time

### 4. Training Benchmark ✅

**Status**: Working correctly (already configured)

| Operation | Time Range | Notes |
|-----------|------------|-------|
| Gradient computation | 420 µs - 5.5 ms | Varies with batch size |
| Backward pass | 1.15 ms - 10 ms | Scales with model complexity |

**Analysis**:
- Training pipeline performance stable
- Higher variance in training benchmarks expected (GPU scheduling)
- No regressions detected

## Validation Summary

### Issues Resolved
- ✅ **Issue #36**: `fused_lora_bench` now produces output
- ✅ **Issue #34**: `test_metal_layer_norm_metal` confirmed passing (fixed in v1.2.0)

### Quality Metrics
- ✅ All 4 benchmarks executed successfully
- ✅ 5 runs each with statistical analysis
- ✅ Results captured with environment snapshot
- ✅ No performance regressions detected
- ✅ Benchmark variance within acceptable ranges (<5% for most operations)

### Performance Claims Validated
- KV-cache: O(1) access time (11.8 ns)
- Lazy evaluation: <10% overhead for complex operations
- Training: Stable gradient computation and backprop
- Inference: Sub-millisecond forward passes

## Comparison with v1.2.0

No performance changes expected or observed - this is a pure configuration bugfix release.

| Metric | v1.2.0 | v1.2.1 | Change |
|--------|--------|--------|--------|
| Inference (forward pass) | ~480 µs | ~480 µs | None |
| KV-cache access | ~12 ns | ~12 ns | None |
| LoRA unfused | ~23 µs | ~24 µs | <5% variance (normal) |
| Lazy matmul overhead | ~1-8% | ~1-8% | None |

## Conclusion

v1.2.1 successfully resolves the benchmark configuration issues without introducing any performance regressions. All benchmarks now execute correctly and produce valid output for release validation.

### Recommendations
- ✅ **Ready for release** - all quality gates passed
- ✅ **No breaking changes** - purely a configuration fix
- ✅ **Performance validated** - no regressions detected
- ✅ **Documentation updated** - CHANGELOG.md reflects fixes

---

**Full benchmark results**: `benchmark_results/20251211_203943/`  
**Generated**: December 11, 2025  
**Validated by**: Official benchmark runner script

