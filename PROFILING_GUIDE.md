# Profiling Guide: Metal Performance Analysis

This guide explains how to profile metal-candle to identify Metal backend bottlenecks.

## Prerequisites

1. **macOS** with Xcode installed
2. **Xcode Instruments** (included with Xcode)
3. **cargo-instruments** (optional, for command-line profiling)

```bash
# Optional: Install cargo-instruments
cargo install cargo-instruments
```

## Quick Start

### Option 1: Using cargo-instruments (Command Line)

```bash
# Profile with Metal System Trace
cargo instruments -t "Metal System Trace" --release --example profile_benchmark

# Profile with Time Profiler
cargo instruments -t "Time Profiler" --release --example profile_benchmark

# Profile with Allocations
cargo instruments -t "Allocations" --release --example profile_benchmark
```

### Option 2: Using Xcode Instruments (GUI)

1. Build the release binary:
   ```bash
   cargo build --release --example profile_benchmark
   ```

2. Open Instruments:
   ```bash
   open -a Instruments
   ```

3. Select a template:
   - **Metal System Trace** (recommended) - GPU analysis
   - **Time Profiler** - CPU analysis
   - **Allocations** - Memory analysis

4. Choose target: `./target/release/examples/profile_benchmark`

5. Click Record ▶️

## What to Look For

### Metal System Trace

#### 1. **Kernel Launch Overhead**

**Question**: How much time is spent launching kernels vs actual GPU work?

**Where to Look**:
- Timeline view → Metal section
- Look for gaps between kernel executions
- Compare kernel execution time to gaps

**What We Expect to See**:
- **MLX**: Tight kernels, minimal gaps
- **metal-candle (current)**: Large gaps, kernel overhead dominates

**Diagnosis**:
```
Good (MLX-like):
  |███|███|███|███|███|  ← Tight kernels, minimal gaps
  
Bad (Current):
  |█|   |█|   |█|   |█|  ← Short kernels, large gaps = overhead
```

#### 2. **Memory Transfer Patterns**

**Question**: Are we transferring data unnecessarily between CPU and GPU?

**Where to Look**:
- Timeline → Memory transfers
- Red flags: Frequent small transfers

**What to Look For**:
- Excessive CPU→GPU or GPU→CPU transfers
- Synchronization points forcing data movement

#### 3. **GPU Utilization**

**Question**: Is the GPU actually busy, or waiting?

**Where to Look**:
- GPU timeline
- Calculate: (GPU busy time) / (total time)

**Good**: >80% utilization  
**Current (suspected)**: <30% utilization due to overhead

#### 4. **Synchronization Points**

**Question**: Are we forcing unnecessary CPU/GPU synchronization?

**Where to Look**:
- Look for `synchronize()` calls
- Timeline gaps where GPU waits for CPU

**What to Fix**:
- Remove unnecessary `.to_vec()` or data reads
- Batch operations before synchronizing

### Time Profiler (CPU Side)

**What to Look For**:
1. Time spent in Metal API calls
2. Overhead in Candle's Metal backend
3. Memory allocation patterns

**Key Functions to Watch**:
- `matmul`
- `softmax`
- `layer_norm`
- `rms_norm`

### Allocations

**What to Look For**:
1. Excessive tensor allocations
2. Memory leaks
3. Fragmentation

## Interpreting Results

### Scenario 1: Kernel Launch Overhead Dominates

**Symptoms**:
- Many small kernel executions
- Large gaps between kernels
- GPU utilization <30%

**Causes**:
- Each operation launches separate kernel
- No operation fusion
- No batching

**Solutions**:
- Batch operations
- Fuse operations (requires Candle changes or custom kernels)
- Increase workload size

### Scenario 2: Memory Transfer Bottleneck

**Symptoms**:
- Frequent CPU↔GPU transfers
- Slow with small tensors

**Causes**:
- Unnecessary synchronization
- Implicit data movement
- Non-contiguous tensors requiring copies

**Solutions**:
- Keep data on GPU longer
- Use `.contiguous()` proactively
- Minimize CPU/GPU synchronization

### Scenario 3: Inefficient Kernels

**Symptoms**:
- GPU is busy but slow
- High utilization but poor throughput

**Causes**:
- Unoptimized Metal kernels in Candle
- Poor memory access patterns
- Lack of kernel-specific optimizations

**Solutions**:
- Custom Metal kernels for critical operations
- Contribute optimizations to Candle upstream
- Use alternative framework (MLX, burn)

## Comparison Framework

### Baseline Metrics

Run the profiling benchmark and record:

```markdown
| Metric | Value | Target (MLX) |
|--------|-------|--------------|
| Total Time | ____ ms | ____ ms |
| GPU Utilization | ___% | >80% |
| Kernel Count | ____ | ____ |
| Avg Kernel Time | ____ µs | ____ µs |
| Avg Gap Time | ____ µs | <1 µs |
```

### Key Performance Indicators

1. **Kernel Launch Efficiency**
   ```
   Efficiency = (Total GPU Compute Time) / (Total Runtime)
   ```
   - Target: >70%
   - Current (estimated): <30%

2. **Operation Throughput**
   ```
   Throughput = (Operations per second)
   ```
   - Compare to MLX baseline

3. **Memory Efficiency**
   ```
   Memory BW = (Bytes transferred) / (Time)
   ```
   - Compare to Metal theoretical max

## Profiling Workflow

### Step 1: Establish Baseline

```bash
# Run profiling benchmark
cargo instruments -t "Metal System Trace" --release --example profile_benchmark

# Save results
mv *.trace mlx_candle_baseline_$(date +%Y%m%d).trace
```

### Step 2: Identify Bottlenecks

Using Instruments GUI:
1. Open the trace file
2. Go to Metal System Trace
3. Sort by GPU time
4. Identify top time consumers

### Step 3: Document Findings

Create `PROFILING_RESULTS.md`:

````markdown
# Profiling Results

## Date: YYYY-MM-DD

### Configuration
- Device: Apple M[X] [Pro/Max/Ultra]
- Candle Version: 0.9.1
- metal-candle Version: 0.1.0

### Key Findings

1. **Bottleneck Identified**: [e.g., Kernel launch overhead]
   - Evidence: [Screenshots/data]
   - Impact: [X%] of total time
   - Root cause: [Analysis]

2. **GPU Utilization**: [X]%
   - Expected: >80%
   - Gap: [Analysis]

3. **Memory Patterns**:
   - Transfers: [X] MB total
   - Bandwidth: [X] GB/s (vs [Y] GB/s theoretical)

### Optimization Opportunities

1. [Specific opportunity]
   - Potential gain: [X]%
   - Effort: [Low/Medium/High]
   - Approach: [Details]

...
````

### Step 4: Test Optimizations

After each optimization:
```bash
cargo bench --bench mlx_comparison
```

Document improvements in `PROFILING_RESULTS.md`.

## Common Issues & Solutions

### Issue 1: Profiling Hangs

**Problem**: Instruments hangs or crashes

**Solution**:
```bash
# Reduce iterations in profile_benchmark.rs
# Change 1000 to 100 for initial profiling
for _ in 0..100 {  // Reduced from 1000
    let _ = layer.forward(&x)?;
}
```

### Issue 2: Can't See Metal Kernels

**Problem**: Timeline shows no Metal activity

**Solution**:
- Ensure you're using Metal device, not CPU
- Check that `Device::new_metal(0)` succeeded
- Verify Metal is available: `metal-rs` examples

### Issue 3: Too Much Data

**Problem**: Trace file is huge, Instruments slow

**Solution**:
- Reduce iterations in benchmark
- Focus on one operation type at a time
- Use shorter profiling runs

## Next Steps After Profiling

Based on profiling results:

### If Kernel Overhead Dominates (Expected)

**Path A: Optimize Candle Usage**
- Batch operations
- Reduce kernel launches
- Improve memory patterns
- Target: 70% of MLX

**Path B: Custom Metal Kernels**
- Write optimized kernels for bottlenecks
- Use `metal-rs` directly
- Target: 95-100% of MLX

**Path C: Alternative Framework**
- MLX Rust bindings
- burn framework
- Target: 100% of MLX (guaranteed)

### If Memory Transfer Bottleneck

**Solutions**:
- Keep tensors on GPU
- Minimize synchronization
- Use in-place operations where possible

### If Kernel Inefficiency

**Solutions**:
- Custom Metal shaders
- Contribute to Candle
- Switch to optimized framework

## Resources

- [Metal Performance Guide](https://developer.apple.com/metal/metal-programming-guide.pdf)
- [Xcode Instruments Guide](https://help.apple.com/instruments/)
- [Candle Documentation](https://github.com/huggingface/candle)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

## Benchmark Code

The `profile_benchmark.rs` example runs:
1. LoRA forward passes (3 sizes)
2. Layer operations (softmax, layer norm, RMS norm)
3. LoRA rank scaling (5 ranks)

Each operation runs 1000 iterations for statistical significance.

---

**Next**: After profiling, proceed to Issue #20 (Optimization) with data-driven approach.

