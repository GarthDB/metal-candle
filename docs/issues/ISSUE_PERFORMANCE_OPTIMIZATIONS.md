# Performance Optimizations (v1.3.1)

## Overview

Conditional performance optimizations based on findings from streaming benchmarks and adapter hot-swapping implementation.

**Priority**: 🟢 Medium (conditional on benchmark findings)  
**Estimate**: 2-3 days  
**Target Release**: v1.3.1 (Late January 2025)

---

## Background

This is a **conditional issue** that should only be worked on if:
1. Streaming benchmarks reveal >5% overhead
2. ApplyAdapter implementation shows >5ms switching time
3. Profiling identifies clear optimization opportunities

---

## Objectives

1. Address performance bottlenecks identified in benchmarks
2. Optimize hot paths discovered during profiling
3. Reduce allocations in critical code paths
4. Improve adapter switching efficiency
5. Maintain code quality and readability

---

## Potential Optimization Areas

### 1. Streaming Inference Path

**Symptoms**:
- Streaming overhead >5%
- High allocation rate
- Callback overhead significant

**Potential Optimizations**:
- [ ] Reduce `StreamToken` allocations
- [ ] Cache tokenizer decoding
- [ ] Optimize probability calculations
- [ ] Reduce string allocations in metadata
- [ ] Use `SmallVec` or stack allocation where possible

**Example**:
```rust
// Before: Allocates on every token
pub struct StreamToken {
    pub token_id: u32,
    pub text: Option<String>,  // Allocation
    pub probability: f32,
    pub logit: f32,
    pub is_eos: bool,
}

// After: Use Cow or stack buffer
pub struct StreamToken<'a> {
    pub token_id: u32,
    pub text: Option<Cow<'a, str>>,  // Zero-copy when possible
    pub probability: f32,
    pub logit: f32,
    pub is_eos: bool,
}
```

### 2. Adapter Switching

**Symptoms**:
- Adapter switching >5ms
- Memory allocation spikes
- GPU synchronization delays

**Potential Optimizations**:
- [ ] Pre-allocate adapter buffers
- [ ] Batch adapter application to layers
- [ ] Reduce GPU/CPU synchronization points
- [ ] Use Metal async compute where possible
- [ ] Cache adapter state

**Example**:
```rust
// Before: Apply adapters one at a time
for layer in &mut self.layers {
    layer.apply_adapter(adapter)?;  // Sync point
}

// After: Batch application
let adapter_data = adapter.prepare_for_batch()?;
self.layers.par_iter_mut()  // Parallel
    .try_for_each(|layer| {
        layer.apply_adapter_batched(&adapter_data)
    })?;
```

### 3. Memory Management

**Symptoms**:
- High peak memory usage
- Frequent allocations
- Long GC pauses

**Potential Optimizations**:
- [ ] Use object pools for temporary tensors
- [ ] Explicit `drop()` for large tensors
- [ ] Arena allocation for related objects
- [ ] Reduce clones, prefer borrows

### 4. Metal GPU Utilization

**Symptoms**:
- Low GPU utilization
- CPU bottleneck
- Inefficient kernels

**Potential Optimizations**:
- [ ] Optimize threadgroup sizes
- [ ] Batch operations where possible
- [ ] Use async compute for overlapping
- [ ] Profile with Metal System Trace

---

## Scope

### In Scope

- Optimizations identified through profiling
- Changes that maintain API compatibility
- Performance improvements without code complexity explosion
- Documentation of trade-offs

### Out of Scope

- Breaking API changes (save for v2.0.0)
- Algorithmic changes (keep for separate issues)
- Features beyond performance (separate issues)
- Premature optimization without measurement

---

## Success Criteria

- ✅ Streaming overhead reduced to <5% (if it was higher)
- ✅ Adapter switching <5ms (if it was slower)
- ✅ No regressions in other benchmarks
- ✅ Code quality maintained (no "clever" code)
- ✅ All tests still passing
- ✅ Documentation updated with optimizations
- ✅ Performance improvements validated by benchmarks

---

## Implementation Plan

### Phase 1: Analysis (1 day)

**Tasks**:
- [ ] Review streaming benchmark results
- [ ] Review ApplyAdapter performance
- [ ] Analyze profiling data from Instruments
- [ ] Identify top 3-5 bottlenecks
- [ ] Estimate impact of each optimization
- [ ] Prioritize by impact/effort ratio

**Deliverables**:
- Performance analysis report
- Prioritized optimization list

### Phase 2: Implementation (1-2 days)

**Tasks**:
- [ ] Implement highest-priority optimizations
- [ ] Add micro-benchmarks for each optimization
- [ ] Verify improvements with benchmarks
- [ ] Ensure no regressions
- [ ] Code review for maintainability

**Deliverables**:
- Optimized code
- Performance validation

### Phase 3: Documentation (0.5 days)

**Tasks**:
- [ ] Document optimization techniques used
- [ ] Update performance documentation
- [ ] Add code comments explaining trade-offs
- [ ] Update CHANGELOG with improvements

**Deliverables**:
- Updated documentation
- Performance comparison

---

## Optimization Guidelines

### Follow These Principles

1. **Measure First**: Always profile before optimizing
2. **Benchmark Everything**: Verify improvements empirically
3. **Keep It Simple**: Avoid "clever" code
4. **Document Trade-offs**: Explain why optimizations are needed
5. **No Regressions**: Test all existing benchmarks

### Avoid These

- Premature optimization without measurement
- Sacrificing readability for marginal gains
- Breaking API compatibility
- Unsafe code without justification
- Optimization that adds complexity

---

## Benchmark Validation

All optimizations must be validated with:

```bash
# Run streaming benchmarks
cargo bench --bench streaming

# Run adapter benchmarks
cargo bench --bench fused_lora_bench

# Run full benchmark suite
cargo bench

# Profile with instruments
cargo instruments -t Time --release --example streaming_demo
```

---

## Example Optimizations

### 1. Reduce StreamToken Allocations

**Before**:
```rust
let token = StreamToken {
    token_id,
    text: Some(tokenizer.decode(&[token_id])?),  // Allocates String
    probability,
    logit,
    is_eos,
};
```

**After**:
```rust
// Pre-allocate decode buffer in Generator
struct Generator {
    decode_buffer: RefCell<String>,
    // ...
}

let token = {
    let mut buf = self.decode_buffer.borrow_mut();
    buf.clear();
    tokenizer.decode_into(&[token_id], &mut buf)?;
    
    StreamToken {
        token_id,
        text: Some(buf.clone()),  // Single allocation, reused buffer
        probability,
        logit,
        is_eos,
    }
};
```

### 2. Batch Adapter Application

**Before**:
```rust
// Apply adapters sequentially with GPU sync
for layer in &mut self.layers {
    layer.apply_adapter(adapter)?;  // GPU sync per layer
}
```

**After**:
```rust
// Batch application with single GPU sync
let adapter_weights: Vec<_> = self.layers.iter()
    .map(|layer| layer.prepare_adapter_weights(adapter))
    .collect();

// Single GPU upload
let gpu_weights = upload_batch_to_gpu(&adapter_weights)?;

// Apply in parallel
self.layers.par_iter_mut()
    .zip(&gpu_weights)
    .try_for_each(|(layer, weights)| {
        layer.apply_adapter_from_gpu(weights)
    })?;
```

---

## Files to Modify

Depends on optimization areas, potentially:

```
src/inference/generator.rs        # Streaming optimizations
src/inference/sampling.rs         # Token generation optimizations
src/models/qwen.rs               # Adapter application optimizations
src/training/adapter_registry.rs  # Registry optimizations
src/backend/metal_ops.rs         # Metal kernel optimizations
```

---

## Performance Targets

### Streaming

- **Current**: TBD from benchmarks
- **Target**: <5% overhead for sync streaming
- **Target**: <10% overhead for async streaming

### Adapter Switching

- **Current**: ~1.3ms (measured)
- **Target**: <1ms if possible, maintain <5ms

### Memory

- **Target**: No increase in peak memory
- **Target**: Reduce allocations per token

---

## Risks & Mitigations

### Risk: Premature Optimization
**Impact**: Wasted effort  
**Mitigation**: Only optimize based on measured bottlenecks

### Risk: Increased Code Complexity
**Impact**: Harder to maintain  
**Mitigation**: Keep code simple, document trade-offs

### Risk: Introducing Bugs
**Impact**: Regressions  
**Mitigation**: Comprehensive testing, benchmark validation

---

## Follow-Up Work

If major optimizations are needed:
- Consider more aggressive optimizations for v1.4.0
- Custom Metal kernels for hot paths
- Algorithm changes (separate issues)

---

## References

- Issue #49: ApplyAdapter implementation
- Issue #50 (TBD): Streaming benchmarks
- ROADMAP.md: v1.3.1 planning
- BENCHMARKS.md: Current performance baseline

---

## Acceptance Checklist

- [ ] Performance analysis completed
- [ ] Top bottlenecks identified
- [ ] Optimizations implemented
- [ ] Benchmarks show improvement
- [ ] No regressions in existing benchmarks
- [ ] Code quality maintained
- [ ] Documentation updated
- [ ] Tests passing
- [ ] PR reviewed and merged

---

**Note**: This issue should only be worked on **after** streaming benchmarks are complete and if optimization is needed.

---

**Created**: December 18, 2024  
**Target**: v1.3.1 (Late January 2025)  
**Depends On**: Streaming benchmarks issue  
**Related Issues**: #49 (ApplyAdapter)

