# MPS Performance Deep Dive

**Date**: December 10, 2024  
**Status**: Command Queue Pooling Implemented - Still 148x Slower

---

## Current Performance

| Implementation | 64×64 Time | vs Candle |
|----------------|------------|-----------|
| **Candle Metal** | 1.65 µs | 1.0x (baseline) |
| **MPS (original)** | 335 µs | 203x SLOWER |
| **MPS (pooled queue)** | 245 µs | 148x SLOWER |
| **Target (MLX)** | ~1-2 µs | Competitive |

**Progress**: Queue pooling gave us 1.4x improvement (90µs saved)  
**Remaining Gap**: Still 148x slower than Candle

---

## Overhead Breakdown Analysis

### What We Fixed ✅
- **Command Queue Creation**: ~90µs saved via pooling

### Remaining Overhead (243µs)

**Per-Operation Costs**:
1. **Command Buffer Creation**: `queue.new_command_buffer()` - Est. 50-100µs?
2. **Synchronous Wait**: `wait_until_completed()` - Est. 50-100µs?
3. **MPSMatrixMultiplication Object**: `MPSMatrixMultiplication::new()` - Est. 30-60µs?
4. **MPS Descriptor Creation**: 3x `MPSMatrixDescriptor::new()` - Est. 20-40µs?
5. **MPS Matrix Creation**: 3x `MPSMatrix::new()` - Est. 20-40µs?
6. **Objective-C Message Overhead**: Multiple `msg_send!` calls - Est. 10-20µs?

**Total Estimated Overhead**: ~180-360µs ✅ Matches observed 243µs

---

## Root Cause: Synchronous Execution Model

### How We're Using MPS (SLOW)

```rust
// Every operation does this:
let queue = get_pooled_queue();           // Fast now (cached)
let cmd_buffer = queue.new_command_buffer();  // ~50µs
let matmul = MPSMatrixMultiplication::new(...); // ~50µs
let mps_left = MPSMatrix::new(...);       // ~20µs
let mps_right = MPSMatrix::new(...);      // ~20µs
let mps_output = MPSMatrix::new(...);     // ~20µs

matmul.encode(cmd_buffer, ...);           // Fast
cmd_buffer.commit();                      // Fast
cmd_buffer.wait_until_completed();        // ~50-100µs BLOCKING

// Total: ~210-260µs setup + ~1-5µs actual GPU work
```

### How Candle Uses Metal (FAST)

Candle likely:
1. **Batches Operations**: Multiple ops in one command buffer
2. **Async Execution**: Doesn't wait immediately
3. **Reuses Buffers**: Pools Metal buffers
4. **Minimal Setup**: Uses pre-configured kernels

**Key Insight**: Candle amortizes setup cost across many operations or doesn't wait synchronously.

---

## Why MPS is Inherently Different from Custom Kernels

### Custom Metal Kernels (Fast Path)
```rust
let pipeline = device.get_pipeline("my_kernel");  // Cached
let cmd_encoder = cmd_buffer.compute_encoder();  // Lightweight
cmd_encoder.set_pipeline_state(pipeline);
cmd_encoder.set_buffer(0, input);
cmd_encoder.dispatch_threads(...);
cmd_encoder.end_encoding();
// No waiting - returns immediately
```

### MPS (High-Level API - More Overhead)
```rust
// MPS does ALL of this internally:
// 1. Validate inputs
// 2. Choose optimal kernel variant
// 3. Configure kernel parameters
// 4. Allocate temporary buffers (if needed)
// 5. Encode multiple internal kernels
// 6. Manage synchronization

// We also do setup:
let matmul = MPSMatrixMultiplication::new(...);
let matrices = create_mps_matrices(...);
matmul.encode(...);
```

**Trade-off**: MPS kernels are more optimized (better compute), but higher invocation overhead.

---

## Solutions (Ranked by Impact)

### Option 1: Remove Synchronous Wait ⚡ HIGH IMPACT

**Current**:
```rust
cmd_buffer.commit();
cmd_buffer.wait_until_completed();  // BLOCKS here
return result;
```

**Proposed**:
```rust
cmd_buffer.commit();
// Don't wait - let Candle sync when needed
return result;  // GPU work happens asynchronously
```

**Pros**:
- Could save 50-100µs
- Enables pipelining
- More similar to Candle's approach

**Cons**:
- Correctness risk (data races)
- Candle might expect synchronous ops
- Harder to debug

**Expected Impact**: 2-3x faster (245µs → 80-120µs)

### Option 2: Pool MPSMatrixMultiplication Objects 🔄 MEDIUM IMPACT

**Observation**: We create a new `MPSMatrixMultiplication` for every operation.

**Proposed**:
```rust
// Cache by (m, n, k) dimensions
static MPS_MATMUL_CACHE: Lazy<DashMap<(usize, usize, usize), MPSMatrixMultiplication>> = ...;

fn get_or_create_matmul(m, n, k) -> &MPSMatrixMultiplication {
    MPS_MATMUL_CACHE.entry((m, n, k)).or_insert_with(|| {
        MPSMatrixMultiplication::new(device, false, false, m, n, k, 1.0, 0.0)
    })
}
```

**Pros**:
- Saves 30-60µs per operation
- Simple caching strategy
- Still correct

**Cons**:
- Memory usage (one object per unique size)
- Cache invalidation complexity
- Device management

**Expected Impact**: 1.2-1.3x faster (245µs → 185-205µs)

### Option 3: Batch Command Encoding 📦 HIGH IMPACT (Long-term)

**Idea**: Encode multiple MPS operations into one command buffer.

**Example**: LoRA has two matmuls: `input @ A` and `hidden @ B`
```rust
let cmd = queue.new_command_buffer();

// Encode both operations
matmul_A.encode(&cmd, input, lora_a, hidden);
matmul_B.encode(&cmd, hidden, lora_b, output);

cmd.commit();
cmd.wait_until_completed();  // One wait for both
```

**Pros**:
- Amortizes command buffer cost
- Natural for LoRA/attention
- Could be 2-3x faster for paired ops

**Cons**:
- Requires API changes
- Only helps multi-op workloads
- Complex synchronization

**Expected Impact**: 2x for LoRA (245µs → 120µs for 2 ops)

### Option 4: Accept MPS is Wrong Tool for Small Ops ❌ LOW IMPACT

**Reality Check**: Maybe MPS isn't meant for tiny, standalone operations.

**MPS Designed For**:
- Large matrices (1024×1024+)
- Batched operations
- High-latency tolerance

**Our Use Case**:
- Small-medium matrices (64×256)
- Single operations
- Low-latency required

**Conclusion**: For our workload, custom kernels might actually be better than MPS.

---

## Recommended Path Forward

### Immediate Actions

1. **Try Async Execution** (1 hour)
   - Remove `wait_until_completed()`
   - Test correctness carefully
   - Measure performance

2. **Document Current State** ✅
   - We've learned a lot
   - MPS might not be the right solution
   - Custom kernels may be better for our workload

### Strategic Decision

**Question**: Should we continue with MPS or pivot back to custom kernels?

**MPS Pros**:
- Theoretically optimal kernels
- Apple-maintained
- No shader development

**MPS Cons**:
- High invocation overhead (~240µs)
- Not designed for single small ops
- Complex caching needed

**Custom Kernels Pros**:
- Lower overhead (~1-5µs)
- Full control
- Already working (37-98µs baseline)

**Custom Kernels Cons**:
- Need optimization work
- Harder to maintain
- Not as fast as MPS for large ops

### Recommendation

**For Small Operations** (64×256): Use custom kernels or Candle  
**For Large Operations** (1024×1024+): MPS might win  
**For LoRA** (mixed sizes): Hybrid approach

---

## Comparison to Original Goals

**Original Goal**: Achieve MLX-level performance (~1-5µs)

**Current Reality**:
- MPS matmul: 245µs
- Candle matmul: 1.65µs
- **Gap**: 148x

**Conclusion**: MPS as currently implemented is **NOT** achieving the goal.

**Options**:
1. Continue optimizing MPS (async, caching, batching)
2. Accept MPS isn't suitable for this workload
3. Use hybrid: Candle for small, MPS for large

---

## Next Steps (User Decision Needed)

### Option A: Continue MPS Optimization
- Implement async execution
- Add matmul object caching
- Target: 50-100µs (still slower than Candle)
- **Timeline**: 2-4 hours more work
- **Success Probability**: Medium (might hit fundamental limits)

### Option B: Pivot to Candle/Custom Hybrid
- Use Candle's Metal matmul (already fast!)
- Optimize custom kernels for specific ops (LoRA, etc.)
- Target: Match or beat Candle
- **Timeline**: Refocus on custom kernel optimization
- **Success Probability**: High (we know this works)

### Option C: Document and Ship Current State
- Document MPS as "available but not optimized"
- Focus on other high-value features
- Revisit when we have large matrix workloads
- **Timeline**: Immediate
- **Success Probability**: N/A (not pursuing performance)

---

## Key Learnings

1. **MPS has high invocation overhead** (~200-300µs)
2. **Command queue pooling helps but isn't enough** (90µs saved)
3. **Candle's Metal backend is already excellent** (1.65µs!)
4. **MPS designed for different workload** (large, batched ops)
5. **Custom kernels may be better for our use case**

---

## Honest Assessment

**MPS Performance for Small Operations**: ❌ Not Competitive  
**Command Queue Pooling**: ✅ Implemented, Modest Improvement  
**Path to MLX Parity**: ⚠️ Unclear, may be architectural mismatch

**Recommendation**: Consult with user on whether to continue MPS path or pivot.

🤔 **The right tool for the job might not be MPS for this workload.**

