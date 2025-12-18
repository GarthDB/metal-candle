# Streaming Performance Benchmarks (v1.3.1)

## Overview

Comprehensive benchmark suite to validate streaming inference performance claims made in v1.3.0.

**Priority**: 🟡 High (validates v1.3.0 claims)  
**Estimate**: 3-5 days  
**Target Release**: v1.3.1 (Late January 2025)

---

## Background

In v1.3.0, we introduced enhanced streaming inference with sync and async APIs. We claimed <5% overhead based on design analysis, but haven't yet measured it empirically. This issue tracks the creation of a comprehensive benchmark suite to validate these claims.

---

## Objectives

1. Measure streaming overhead vs non-streaming generation
2. Compare sync vs async streaming performance
3. Profile callback overhead per token
4. Identify and document any performance bottlenecks
5. Validate or adjust performance claims in documentation

---

## Scope

### In Scope

#### 1. Benchmark Suite Creation

**File**: `benches/streaming.rs`

Benchmarks to implement:
- Baseline: `generate()` (non-streaming)
- Sync streaming: `generate_stream()` with callback
- Async streaming: `generate_stream_async()` with stream consumption
- Various buffer sizes (if applicable)
- Different model sizes (if available)

#### 2. Metrics to Measure

- **Throughput**: Tokens per second
- **Latency**: Time per token
- **Overhead**: Percentage difference vs baseline
- **Memory**: Peak memory usage
- **Callback Cost**: Time spent in callback vs generation

#### 3. Profiling

- Use `cargo instruments` for detailed profiling
- Identify hot paths in streaming code
- Measure allocation overhead
- Profile async runtime overhead

#### 4. Documentation

- Update `CHANGELOG.md` with actual measurements
- Document results in `BENCHMARKS.md`
- Update API docs with performance characteristics
- Create performance comparison tables

### Out of Scope

- Performance optimizations (separate issue)
- KV-cache benchmarks (already covered elsewhere)
- Training benchmarks (separate concern)

---

## Expected Results

Based on design analysis, we expect:

```
generate() (baseline):       1000 tokens in 2.50s (400 tok/s)
generate_stream() (sync):    1000 tokens in 2.60s (385 tok/s) [+4% overhead]
generate_stream_async():     1000 tokens in 2.65s (377 tok/s) [+6% overhead]
```

**Target**: <5% overhead for sync streaming, <10% for async

---

## Success Criteria

- ✅ Comprehensive benchmark suite in `benches/streaming.rs`
- ✅ Baseline measurements documented
- ✅ Streaming overhead <5% (or documented if higher)
- ✅ Async overhead <10% (or documented if higher)
- ✅ Performance results integrated into CI (optional)
- ✅ Documentation updated with actual measurements
- ✅ Profiling results analyzed and documented

---

## Implementation Plan

### Phase 1: Benchmark Creation (2 days)

**Tasks**:
- [ ] Create `benches/streaming.rs` file
- [ ] Implement baseline benchmark (`generate()`)
- [ ] Implement sync streaming benchmark (`generate_stream()`)
- [ ] Implement async streaming benchmark (`generate_stream_async()`)
- [ ] Add criterion configuration for consistent results
- [ ] Test benchmarks on M4 Max

**Deliverables**:
- Working benchmark suite
- Repeatable benchmark execution

### Phase 2: Measurement & Analysis (1 day)

**Tasks**:
- [ ] Run benchmarks multiple times (5+ runs)
- [ ] Collect consistent measurements
- [ ] Calculate statistical significance
- [ ] Profile with Instruments.app
- [ ] Identify any unexpected overhead
- [ ] Document findings

**Deliverables**:
- Benchmark results spreadsheet
- Profiling reports
- Performance analysis document

### Phase 3: Documentation (1 day)

**Tasks**:
- [ ] Update `CHANGELOG.md` with measurements
- [ ] Add streaming benchmarks section to `BENCHMARKS.md`
- [ ] Update API docs with performance notes
- [ ] Create performance comparison table
- [ ] Document any caveats or limitations

**Deliverables**:
- Updated documentation
- Performance recommendations

### Phase 4: Integration (Optional, 1 day)

**Tasks**:
- [ ] Add streaming benchmarks to CI
- [ ] Set up baseline tracking
- [ ] Create regression detection
- [ ] Document CI integration

**Deliverables**:
- CI-integrated benchmarks
- Automated regression detection

---

## Technical Details

### Benchmark Structure

```rust
// benches/streaming.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use metal_candle::{
    models::ModelLoader,
    inference::{Generator, GenerationConfig},
};

fn streaming_benchmarks(c: &mut Criterion) {
    // Setup
    let model = load_test_model();
    let input_ids = create_test_input();
    let config = GenerationConfig::default()
        .with_max_length(100);
    
    let mut generator = Generator::new(model, config);
    
    // Baseline: Non-streaming
    c.bench_function("generate_baseline", |b| {
        b.iter(|| {
            generator.generate(&input_ids)
        });
    });
    
    // Sync streaming
    c.bench_function("generate_stream_sync", |b| {
        b.iter(|| {
            generator.generate_stream(&input_ids, |token| {
                // Minimal callback
                !token.is_eos
            })
        });
    });
    
    // Async streaming
    c.bench_function("generate_stream_async", |b| {
        b.to_async(Runtime::new().unwrap())
         .iter(|| async {
            let mut stream = generator.generate_stream_async(&input_ids).unwrap();
            while let Some(token) = stream.next().await {
                let _ = token;
            }
        });
    });
}

criterion_group!(benches, streaming_benchmarks);
criterion_main!(benches);
```

### Profiling Commands

```bash
# CPU profiling
cargo instruments -t Time --release --bench streaming

# Memory profiling
cargo instruments -t Allocations --release --bench streaming

# Metal GPU profiling
cargo instruments -t Metal --release --bench streaming
```

---

## Files to Modify

```
benches/streaming.rs              # New benchmark file
CHANGELOG.md                      # Update with measurements
BENCHMARKS.md                     # Add streaming section
src/inference/generator.rs        # Add performance notes to docs
docs/STREAMING_PERFORMANCE.md     # New detailed performance doc (optional)
```

---

## Dependencies

- `criterion` - Already in dev-dependencies
- `tokio` - For async benchmarks (if not already present)
- Test model/data for consistent benchmarking

---

## Risks & Mitigations

### Risk: Benchmark Variance
**Impact**: Inconsistent results  
**Mitigation**: Multiple runs, statistical analysis, controlled environment

### Risk: Overhead Higher Than Expected
**Impact**: Need to adjust claims  
**Mitigation**: Document actual findings, create optimization issue

### Risk: Async Overhead Significant
**Impact**: Users may avoid async API  
**Mitigation**: Document trade-offs, provide guidance on when to use each

---

## Follow-Up Work

If benchmarks reveal issues:
- Create optimization issue (separate)
- Prioritize based on severity
- Document workarounds in the meantime

---

## References

- v1.3.0 PR #48: Original streaming implementation
- CHANGELOG.md: Current performance claims
- ROADMAP.md: v1.3.1 planning

---

## Acceptance Checklist

- [ ] Benchmark suite created and tested
- [ ] Baseline measurements collected
- [ ] Streaming overhead measured (<5% target)
- [ ] Async overhead measured (<10% target)
- [ ] Results documented in CHANGELOG
- [ ] BENCHMARKS.md updated
- [ ] API docs updated with performance notes
- [ ] Profiling completed and analyzed
- [ ] CI integration (optional)
- [ ] PR reviewed and merged

---

**Created**: December 18, 2024  
**Target**: v1.3.1 (Late January 2025)  
**Related Issues**: #49 (ApplyAdapter)

