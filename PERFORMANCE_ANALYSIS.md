# Performance Analysis: MLX vs metal-candle

**Date**: October 2025  
**Status**: 🔴 BLOCKER - Performance target not met  
**Gap**: metal-candle is 55% slower than MLX (45% of MLX performance)

## Benchmark Results Summary

### Overall Performance
- **Target**: 90-100% of MLX performance
- **Actual**: 45% of MLX performance
- **Gap**: 55% slower than MLX
- **Status**: ❌ NOT MET

### Detailed Results

#### LoRA Forward Pass
| Operation | MLX (µs) | metal-candle (µs) | Ratio |
|-----------|----------|-------------------|-------|
| Small (512x512, r=8) | 5.45 | 8.98 | 0.61x |
| Medium (1024x1024, r=8) | 4.97 | 9.68 | 0.51x |
| Large (2048x2048, r=8) | 7.25 | 14.69 | 0.49x |

**Analysis**: MLX is 2x faster for LoRA operations. Gap widens with larger matrices.

#### Layer Operations (1024x1024)
| Operation | MLX (µs) | metal-candle (µs) | Ratio |
|-----------|----------|-------------------|-------|
| Softmax | 1.58 | 6.77 | 0.23x |
| Layer Norm | 1.99 | 12.33 | 0.16x |
| RMS Norm | 5.69 | 8.37 | 0.68x |

**Analysis**: Layer operations show 3-6x performance gap. LayerNorm particularly slow.

#### LoRA Rank Scaling
| Rank | MLX (µs) | metal-candle (µs) | Ratio |
|------|----------|-------------------|-------|
| 4 | 4.93 | 7.49 | 0.66x |
| 8 | 5.07 | 8.29 | 0.61x |
| 16 | 6.42 | 5.50 | **1.17x** ✅ |
| 32 | 5.03 | 4.75 | **1.06x** ✅ |
| 64 | 7.19 | 4.86 | **1.48x** ✅ |

**Analysis**: metal-candle becomes **competitive at high ranks** (16+). Suggests fixed overhead dominates small operations.

## Root Cause Analysis

### 1. Candle Metal Backend Characteristics

**Potential Issues:**
- **Dispatch Overhead**: Each Metal operation has kernel launch overhead
- **Memory Management**: Synchronization points may be inefficient
- **Kernel Optimization**: Candle's Metal kernels may not be as optimized as MLX's
- **Graph Optimization**: No lazy evaluation or graph fusion in our usage

**Evidence:**
- Small operations (5-15 µs range) show biggest gap
- High-rank operations competitive (less affected by overhead)
- LayerNorm (complex operation) particularly slow

### 2. MLX Advantages

MLX is **purpose-built for Apple Silicon**:
- Custom Metal kernel implementations
- Lazy evaluation with graph optimization
- Minimal Python/Metal boundary overhead
- Apple collaboration on Metal optimizations
- Unified memory architecture exploitation

### 3. Our Usage of Candle

**Current Approach:**
```rust
// Each operation is a separate kernel launch
let base_output = x.matmul(&weight)?;
let lora_a_output = x.matmul(&lora_a)?;
let lora_b_output = lora_a_output.matmul(&lora_b)?;
let scaled = lora_b_output.mul(scaling)?;
base_output.add(&scaled)?
```

**Issues:**
- 5 separate Metal kernel launches
- No fusion/optimization
- Eager evaluation

**MLX Approach:**
```python
# Lazy evaluation with graph fusion
output = base @ weight + (x @ lora_a @ lora_b) * scaling
# MLX fuses operations before executing
```

## Investigation Areas

### 1. Candle Metal Backend Performance

**Questions:**
- Is Candle Metal backend actively optimized?
- Are there known performance issues?
- Community benchmarks vs MLX?

**Action**: Research Candle GitHub issues and discussions

### 2. Alternative Frameworks

**Candidates for Apple Silicon ML in Rust:**

#### Option A: Continue with Candle + Optimizations
- **Pros**: Already invested, active community, supports Metal
- **Cons**: May have fundamental architectural limitations
- **Approach**: Profile, optimize, contribute upstream

#### Option B: Direct Metal Shaders in Rust
- **Pros**: Full control, maximum performance
- **Cons**: Massive engineering effort, maintenance burden
- **Framework**: `metal-rs` crate
- **Effort**: 3-6 months for core operations

#### Option C: MLX Rust Bindings
- **Pros**: Best performance, battle-tested
- **Cons**: Still depends on C++/Python, not pure Rust
- **Approach**: Create safe Rust bindings to MLX C++ API
- **Effort**: 2-4 weeks

#### Option D: Other Rust ML Frameworks
- **burn**: Multi-backend, supports Metal
  - Status: Newer, less mature than Candle
  - Performance: Unknown vs MLX
- **tract**: ONNX inference
  - Status: Mature, but inference-focused
  - Training: Limited support

### 3. Profiling Our Implementation

**Tools:**
- Xcode Instruments (Metal profiling)
- `cargo flamegraph`
- Candle debug/tracing

**Focus:**
- Where is time spent?
- Metal kernel launch overhead?
- Memory transfer patterns?
- Synchronization points?

## Optimization Opportunities (Short-term)

### 1. Reduce Kernel Launches
```rust
// Try to fuse operations where possible
// Use Candle's graph optimization if available
```

### 2. Batch Operations
```rust
// Group matrix multiplications
// Minimize Metal API calls
```

### 3. Memory Layout
```rust
// Ensure contiguous tensors
// Minimize reshapes
```

### 4. Check Candle Updates
- Upgrade to latest Candle version
- Enable any Metal-specific optimizations
- Use Metal-optimized dtypes

## Decision Framework

### Scenarios

#### Scenario 1: Quick Wins Available (1-2 weeks)
**If** we can get to 70-80% of MLX with optimizations:
- ✅ Continue with Candle
- 📝 Document performance tradeoffs
- 🚀 Proceed to Phase 6
- 📅 Plan v1.1 optimization push

#### Scenario 2: Fundamental Limitations (Current)
**If** Candle has architectural limits:
- 🔄 Evaluate alternatives (MLX bindings, direct Metal)
- 📊 Cost-benefit analysis
- 🎯 Revised timeline (2-4 weeks delay)
- 🤔 Consider hybrid approach

#### Scenario 3: MLX Bindings Most Pragmatic
**If** performance is critical and timeline matters:
- ✅ Best performance guarantee
- ⚠️ Not pure Rust (but single binary still works)
- 📦 Acceptable for production use
- 🚀 Fastest path to v1.0

## Recommendation

### Immediate Actions (This Week)

1. **Profile metal-candle** (1-2 hours)
   - Use Instruments to see Metal kernel times
   - Identify specific bottlenecks
   - Measure overhead vs computation

2. **Research Candle** (2-3 hours)
   - Check recent issues/PRs about Metal performance
   - Look for Metal optimization flags
   - Check if we're using Candle correctly

3. **Prototype MLX Bindings** (4-6 hours)
   - Create minimal Rust wrapper for MLX
   - Benchmark LoRA forward pass
   - Assess feasibility

4. **Compare burn Framework** (2-3 hours)
   - Install and test burn with Metal backend
   - Run same benchmarks
   - Evaluate maturity

### Decision Point (End of Week)

**Based on findings, choose:**
- **Path A**: Optimize Candle (if 70-80% achievable)
- **Path B**: Switch to MLX bindings (if Candle limited)
- **Path C**: Pivot to burn or other framework

## Success Criteria

For v1.0 Release:
- **Minimum**: 70% of MLX performance (acceptable with documentation)
- **Target**: 80-90% of MLX performance
- **Ideal**: 90-100% of MLX performance

## Notes

- MLX is an **exceptional baseline** (Apple-optimized)
- 70-80% of MLX is still **very fast** for production
- Pure Rust benefits may justify small performance gap
- High-rank LoRA performance is **already competitive**

## Next Steps

1. ✅ Document current findings (this file)
2. ⏳ Profile with Instruments
3. ⏳ Research Candle optimizations
4. ⏳ Prototype alternatives
5. ⏳ Make architecture decision
6. ⏳ Update project timeline

---

**Last Updated**: October 2025  
**Owner**: metal-candle team  
**Priority**: 🔴 CRITICAL - BLOCKER

