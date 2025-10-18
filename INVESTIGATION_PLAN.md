# Performance Investigation & Decision Plan

## Current Situation
- **Performance Gap**: metal-candle at 45% of MLX speed (55% slower)
- **Target**: 90-100% of MLX performance
- **Status**: 🔴 BLOCKER for v1.0 release

## This Week's Investigation (Priority Order)

### Day 1-2: Profile & Optimize Current Implementation

#### 1. Profile with Xcode Instruments (2 hours)
```bash
# Build with debug symbols for profiling
cargo instruments -t "Metal System Trace" --release --example train_lora

# Look for:
# - Metal kernel launch overhead
# - Memory transfer patterns
# - Synchronization points
# - Actual GPU compute time vs overhead
```

#### 2. Check Candle Usage Patterns (1 hour)
- Review our tensor operations for inefficiencies
- Check if we're causing unnecessary synchronization
- Verify we're using Metal-optimized operations
- Look for tensor copies or reshape overhead

#### 3. Quick Optimization Attempts (2-3 hours)
- Reduce intermediate tensor allocations
- Batch operations where possible
- Check for contiguous memory access patterns
- Try different dtype (F16 vs F32)

**Decision Point**: Can we reach 70% of MLX with quick wins?
- ✅ YES → Continue with Candle, document tradeoffs
- ❌ NO → Proceed to alternatives evaluation

### Day 3: Evaluate Alternatives

#### Option A: MLX Rust Bindings (RECOMMENDED if Candle limited)

**Pros:**
- Best performance guarantee (proven in benchmarks)
- Battle-tested by Apple/community
- Still produces single binary (static linking)
- Fastest path to v1.0

**Cons:**
- Not pure Rust (wraps C++ MLX library)
- Depends on MLX installation
- Slightly less idiomatic

**Effort**: 2-4 days to create bindings + integrate

**Prototype**:
```rust
// mlx-sys/build.rs - bindgen to MLX C++ API
// metal-candle-mlx/lib.rs - Safe Rust wrapper
use mlx_sys::*;

pub struct MLXLoRALayer {
    // Wraps MLX array and operations
}
```

#### Option B: burn Framework

**Status**: Research needed
- Check Metal backend maturity
- Run same benchmarks
- Compare API surface

**Effort**: 1-2 days evaluation + 1-2 weeks migration if promising

#### Option C: tch-rs (PyTorch Rust Bindings)

**Status**: Mature but heavy dependency
- Already proven performance
- Large ecosystem
- Not Apple Silicon optimized like MLX

**Effort**: 2-3 weeks migration

### Day 4: Prototype Best Alternative

Based on Day 3 findings, implement minimal prototype:
- LoRA layer forward pass
- Basic training step
- Benchmark against MLX

### Day 5: Decision & Path Forward

## Decision Matrix

| Option | Performance | Effort | Risk | Timeline Impact |
|--------|-------------|--------|------|-----------------|
| Optimize Candle | 70-80%? | 1 week | Low | None |
| MLX Bindings | 100% | 2-4 days | Low | +1 week |
| burn Framework | Unknown | 2+ weeks | Medium | +2-3 weeks |
| Direct Metal | 100% | 3+ months | High | +12 weeks |

## Recommendation Path

### Preferred: Pragmatic Approach

1. **Spend 2 days** optimizing current Candle implementation
2. **If < 70% MLX**: Pivot to MLX Rust bindings
3. **Document tradeoffs** clearly
4. **Ship v1.0** with best available option
5. **Plan v1.1** for pure Rust if needed

### Why MLX Bindings Are Acceptable

**The goal was**: Replace PyO3+MLX with pure Rust for single-binary deployment

**MLX Rust bindings achieve**:
- ✅ Single binary (statically linked)
- ✅ No Python runtime needed
- ✅ Best performance
- ✅ Rust API for consumers
- ⚠️ Internal C++ dependency (acceptable tradeoff)

**This is still a huge win over PyO3**:
- No Python interpreter
- No pip dependencies
- Clean Rust API
- Single binary distribution

## Success Criteria Revision

### v1.0 Release Criteria
- **Minimum**: 70% of MLX (with Candle + docs)
- **Target**: 100% of MLX (with MLX bindings)

### v1.1 Could Explore
- Pure Rust optimization
- Custom Metal kernels
- Graph optimization layer

## Next Actions

1. ✅ Document findings (PERFORMANCE_ANALYSIS.md)
2. ✅ Create investigation plan (this file)
3. ⏳ Profile with Instruments
4. ⏳ Quick optimization pass
5. ⏳ Prototype MLX bindings (if needed)
6. ⏳ Make final decision
7. ⏳ Update PLAN.md with revised approach

---

**Timeline**: 1 week investigation → decision → implementation  
**Impact on v1.0**: +1-2 weeks delay (acceptable)  
**Priority**: 🔴 CRITICAL
