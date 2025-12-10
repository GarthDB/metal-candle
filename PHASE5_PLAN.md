# Phase 5: Async Execution & Performance Optimization

**Date**: December 9, 2024  
**Duration**: Weeks 10-12 (3 weeks)  
**Status**: STARTING  
**Goal**: Achieve 2-3x speedup through async Metal execution and graph optimization

---

## Overview

Phase 5 transforms the synchronous lazy evaluation framework (Phase 4) into a high-performance asynchronous system. This phase targets **2-3x speedup** over v1.0 eager execution, bringing `metal-candle` to 50-95% of MLX performance.

---

## Current State (Phase 4 Baseline)

### Performance

| Operation | v1.0 Eager | Phase 4 Lazy | Overhead |
|-----------|-----------|-------------|----------|
| LoRA | 36 µs | 38 µs | +5.6% |
| Softmax | 39 µs | 41 µs | +5.1% |
| RMS Norm | 47 µs | 49 µs | +4.3% |

### Architecture

- ✅ LazyTensor API - Defers execution
- ✅ ComputationGraph - DAG with topological sort
- ✅ AsyncExecutor - **Currently synchronous**
- ✅ Operation enum - Basic operations
- ⚠️ No optimization passes
- ⚠️ No async command buffer batching
- ⚠️ No operation fusion

---

## Phase 5 Goals

### Week 10: Async Metal Execution

**Goal**: Batch Metal command buffers for parallel execution

**Tasks**:
1. Implement async command buffer queue
2. Add Metal command encoder pooling
3. Batch graph operations into single command buffer
4. Profile with Instruments (Metal)

**Expected Outcome**: 20-30% speedup from reduced kernel launch overhead

### Week 11: Graph Optimization

**Goal**: Fuse operations for better performance

**Tasks**:
1. Implement operation fusion passes
2. Add LoRA fusion (input @ A @ B → single kernel)
3. Add matmul + bias fusion
4. Add activation fusion (ReLU, GELU after matmul)

**Expected Outcome**: 40-60% speedup from reduced memory transfers

### Week 12: Profiling & Optimization

**Goal**: Identify and fix bottlenecks

**Tasks**:
1. Profile with Instruments (Time + Metal)
2. Optimize graph overhead
3. Tune Metal kernel parameters
4. Benchmark vs MLX

**Expected Outcome**: Hit 2-3x overall speedup target

---

## Detailed Implementation Plan

### Week 10: Async Metal Execution

#### 1. Metal Command Buffer Queue

**File**: `src/graph/executor.rs`

```rust
use metal::{CommandQueue, CommandBuffer};
use std::collections::VecDeque;

pub struct AsyncExecutor {
    device: Device,
    command_queue: Arc<CommandQueue>,
    pending_buffers: VecDeque<CommandBuffer>,
    max_batch_size: usize,
}

impl AsyncExecutor {
    pub fn new(device: Device) -> Self {
        let command_queue = device.metal_device()
            .unwrap()
            .new_command_queue();
        
        Self {
            device,
            command_queue: Arc::new(command_queue),
            pending_buffers: VecDeque::new(),
            max_batch_size: 16,
        }
    }
    
    /// Execute graph asynchronously
    pub async fn execute_async(&mut self, graph: &ComputationGraph) -> Result<()> {
        // 1. Batch operations into command buffer
        let cmd_buffer = self.command_queue.new_command_buffer();
        
        // 2. Encode all operations
        for node in graph.topological_order()? {
            self.encode_operation(cmd_buffer, node)?;
        }
        
        // 3. Commit and track
        cmd_buffer.commit();
        self.pending_buffers.push_back(cmd_buffer);
        
        // 4. Flush if batch is full
        if self.pending_buffers.len() >= self.max_batch_size {
            self.flush().await?;
        }
        
        Ok(())
    }
    
    /// Wait for all pending operations
    pub async fn flush(&mut self) -> Result<()> {
        while let Some(buffer) = self.pending_buffers.pop_front() {
            buffer.wait_until_completed();
        }
        Ok(())
    }
}
```

**Benefits**:
- Multiple operations in single command buffer
- Reduced CPU-GPU synchronization
- Better GPU utilization

#### 2. Operation Batching

**File**: `src/graph/executor.rs`

```rust
impl AsyncExecutor {
    fn encode_operation(
        &self,
        cmd_buffer: &CommandBuffer,
        node: &GraphNode,
    ) -> Result<()> {
        let encoder = cmd_buffer.new_compute_command_encoder();
        
        match &node.operation {
            Operation::Matmul { .. } => {
                self.encode_matmul(encoder, node)?;
            }
            Operation::LoRA { .. } => {
                self.encode_lora(encoder, node)?;
            }
            Operation::Softmax { .. } => {
                self.encode_softmax(encoder, node)?;
            }
            // ... other operations
        }
        
        encoder.end_encoding();
        Ok(())
    }
}
```

#### 3. Benchmarking

**File**: `benches/async_execution.rs`

```rust
fn benchmark_async_vs_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_execution");
    
    // Sync (Phase 4)
    group.bench_function("sync_lora_chain", |b| {
        b.iter(|| {
            let output1 = lora1.forward_lazy(&input)?.eval()?;
            let output2 = lora2.forward_lazy(&output1)?.eval()?;
            black_box(output2);
        });
    });
    
    // Async (Phase 5)
    group.bench_function("async_lora_chain", |b| {
        b.iter(|| {
            // Build full graph
            let output1 = lora1.forward_lazy(&input)?;
            let output2 = lora2.forward_lazy(&output1)?;
            
            // Single async execution
            let result = output2.eval_async()?.await?;
            black_box(result);
        });
    });
}
```

**Target**: 20-30% speedup from batching

---

### Week 11: Graph Optimization

#### 1. Fusion Pass Framework

**File**: `src/graph/optimizer.rs` (new)

```rust
pub trait FusionPass {
    fn name(&self) -> &str;
    fn try_fuse(&self, graph: &mut ComputationGraph, node: NodeId) -> Result<bool>;
}

pub struct GraphOptimizer {
    passes: Vec<Box<dyn FusionPass>>,
}

impl GraphOptimizer {
    pub fn new() -> Self {
        Self {
            passes: vec![
                Box::new(LoRAFusion),
                Box::new(MatmulBiasFusion),
                Box::new(ActivationFusion),
            ],
        }
    }
    
    pub fn optimize(&self, graph: &mut ComputationGraph) -> Result<()> {
        for pass in &self.passes {
            let mut changed = true;
            while changed {
                changed = false;
                for node_id in graph.nodes() {
                    if pass.try_fuse(graph, node_id)? {
                        changed = true;
                        break;
                    }
                }
            }
        }
        Ok(())
    }
}
```

#### 2. LoRA Fusion Pass

**File**: `src/graph/optimizer.rs`

```rust
struct LoRAFusion;

impl FusionPass for LoRAFusion {
    fn name(&self) -> &str {
        "lora_fusion"
    }
    
    fn try_fuse(&self, graph: &mut ComputationGraph, node: NodeId) -> Result<bool> {
        // Detect pattern: input @ A @ B * scale
        // Replace with: FusedLoRA(input, A, B, scale)
        
        let node = graph.get_node(node)?;
        
        // Check if this is a scale operation
        if let Operation::MulScalar { input, scale } = &node.operation {
            // Check if input is matmul
            if let Some(matmul_node) = graph.get_node(*input).ok() {
                if let Operation::Matmul { left, right } = &matmul_node.operation {
                    // Check if left is also matmul
                    if let Some(first_matmul) = graph.get_node(*left).ok() {
                        if matches!(first_matmul.operation, Operation::Matmul { .. }) {
                            // Found pattern! Fuse into single LoRA operation
                            graph.fuse_nodes(
                                vec![first_matmul.id, matmul_node.id, node.id],
                                Operation::LoRA {
                                    // ... construct fused op
                                },
                            )?;
                            return Ok(true);
                        }
                    }
                }
            }
        }
        
        Ok(false)
    }
}
```

#### 3. Testing Fusion

**File**: `tests/graph/optimizer.rs`

```rust
#[test]
fn test_lora_fusion() -> Result<()> {
    let device = Device::Cpu;
    let graph = ComputationGraph::new(device.clone());
    
    // Build unfused graph
    let input = LazyTensor::from_slice(&[1.0; 512], &[512], &device)?;
    let a = LazyTensor::from_slice(&[...], &[512, 8], &device)?;
    let b = LazyTensor::from_slice(&[...], &[8, 512], &device)?;
    
    let hidden = input.matmul(&a)?;
    let output = hidden.matmul(&b)?;
    let scaled = output.mul_scalar(2.0)?;
    
    // Graph should have 4 nodes: input, a, b, matmul, matmul, scale
    assert_eq!(scaled.graph_size(), 6);
    
    // Optimize
    let optimizer = GraphOptimizer::new();
    optimizer.optimize(&mut graph)?;
    
    // After fusion, should have fewer nodes
    assert!(scaled.graph_size() < 6);
    
    // Verify correctness unchanged
    let result = scaled.eval()?;
    // ... compare with unfused
    
    Ok(())
}
```

**Target**: 40-60% speedup from fusion

---

### Week 12: Profiling & Optimization

#### 1. Instruments Profiling

**Commands**:
```bash
# Time profiler
cargo instruments -t Time --release --example train_lora

# Metal profiler
cargo instruments -t Metal --release --example train_lora

# Allocations profiler
cargo instruments -t Allocations --release --example train_lora
```

**Focus Areas**:
- Graph building overhead
- Metal kernel dispatch latency
- Memory allocations in hot paths
- Command buffer commit frequency

#### 2. Graph Overhead Optimization

**Current Issue**: Each `LazyTensor` operation acquires locks

**Solution**: Lock-free graph building with atomic operations

```rust
pub struct ComputationGraph {
    nodes: Arc<RwLock<HashMap<NodeId, GraphNode>>>,  // Current
    next_node_id: AtomicUsize,
}

// Optimize to:
pub struct ComputationGraph {
    nodes: DashMap<NodeId, GraphNode>,  // Lock-free concurrent hashmap
    next_node_id: AtomicUsize,
}
```

#### 3. Metal Kernel Tuning

**For each custom kernel**:
- Test different threadgroup sizes
- Optimize shared memory usage
- Tune grid dimensions

**Script**: `scripts/tune_kernels.sh`

```bash
#!/bin/bash
for tg_size in 16 32 64 128 256; do
    echo "Testing threadgroup size: $tg_size"
    cargo bench lora_forward -- --save-baseline "tg_$tg_size"
done

# Compare results
cargo bench lora_forward -- --baseline "tg_16" --load-baseline "tg_256"
```

#### 4. MLX Comparison

**File**: `benches/mlx_comparison_v2.rs`

```rust
// Run metal-candle v2.0 benchmarks
let candle_times = benchmark_candle_async()?;

// Run MLX benchmarks (via Python script)
let mlx_times = benchmark_mlx()?;

// Compare
for op in ["lora", "softmax", "rms_norm"] {
    let ratio = candle_times[op] / mlx_times[op];
    println!("{}: {:.2}x MLX performance", op, 1.0 / ratio);
}
```

**Target**: Achieve 50-95% of MLX performance

---

## Success Metrics

### Performance Targets

| Operation | v1.0 | Phase 4 | Phase 5 Target | MLX |
|-----------|------|---------|----------------|-----|
| LoRA | 36 µs | 38 µs | **10-15 µs** | 5-11 µs |
| Softmax | 39 µs | 41 µs | **5-8 µs** | 1.85 µs |
| RMS Norm | 47 µs | 49 µs | **10-15 µs** | 6 µs |

**Overall**: 2-3x speedup vs v1.0, 50-95% of MLX

### Code Quality

- ✅ Zero clippy errors
- ✅ Comprehensive benchmarks
- ✅ Instruments profiling data
- ✅ MLX comparison documented

### Test Coverage

- ✅ All Phase 4 tests still passing
- ✅ New async execution tests
- ✅ Fusion correctness tests
- ✅ Performance regression tests

---

## Technical Risks & Mitigations

### Risk 1: Async Complexity

**Risk**: Async Metal APIs are complex  
**Mitigation**: Start with simple batching, iterate  
**Fallback**: Keep synchronous path if async doesn't improve performance

### Risk 2: Fusion Bugs

**Risk**: Incorrect fusion breaks correctness  
**Mitigation**: Extensive testing, compare with unfused results  
**Fallback**: Disable specific fusion passes if issues found

### Risk 3: Performance Target Not Met

**Risk**: Don't achieve 2-3x speedup  
**Mitigation**: Profile early, optimize incrementally  
**Fallback**: Document actual performance, adjust expectations

---

## Dependencies

### External Crates

```toml
[dependencies]
# Existing
metal = "0.27"
tokio = { version = "1.0", features = ["rt", "sync"] }  # For async

# New
dashmap = "5.5"  # Lock-free concurrent hashmap
async-trait = "0.1"  # Async traits
```

### Internal Dependencies

- Phase 3: Core graph infrastructure ✅
- Phase 4: Operation migration ✅
- Custom Metal kernels (v1.0) ✅

---

## Timeline

### Week 10 (Days 1-7)
- Day 1-2: Implement command buffer queue
- Day 3-4: Add operation batching
- Day 5-6: Test and benchmark
- Day 7: Document results

### Week 11 (Days 8-14)
- Day 8-9: Fusion pass framework
- Day 10-11: Implement LoRA fusion
- Day 12-13: Test fusion correctness
- Day 14: Benchmark fusion speedup

### Week 12 (Days 15-21)
- Day 15-16: Instruments profiling
- Day 17-18: Optimize bottlenecks
- Day 19-20: MLX comparison
- Day 21: Final documentation

---

## Deliverables

### Code
- `src/graph/executor.rs` - Async execution
- `src/graph/optimizer.rs` - Fusion passes
- `benches/async_execution.rs` - Performance benchmarks
- `tests/graph/optimizer.rs` - Fusion tests

### Documentation
- `PHASE5_WEEK10_COMPLETE.md` - Async execution results
- `PHASE5_WEEK11_COMPLETE.md` - Fusion results
- `PHASE5_COMPLETE.md` - Full Phase 5 summary
- `PROFILING_RESULTS_V2.md` - Instruments data
- `MLX_COMPARISON_V2.md` - Performance comparison

### Benchmarks
- Async vs sync comparison
- Fusion speedup analysis
- MLX performance ratio
- Memory usage profiling

---

## Next Steps After Phase 5

### Phase 6: Documentation & Release (Weeks 13-14)

**Goals**:
1. Update all documentation
2. Create comprehensive examples
3. Write v2.0 release notes
4. Publish to crates.io

**Deliverables**:
- Updated README, ARCHITECTURE.md
- Example code for common use cases
- Migration guide finalized
- v2.0 release published

---

## Summary

Phase 5 transforms `metal-candle` from a correct lazy evaluation implementation into a high-performance ML framework that rivals MLX on Apple Silicon.

**Key Innovations**:
- 🚀 Async Metal command buffer batching
- 🔀 Automatic operation fusion
- 📊 Comprehensive profiling and optimization
- 🎯 2-3x speedup target

**Timeline**: 3 weeks (Weeks 10-12)  
**Outcome**: Production-ready v2.0 with near-MLX performance

---

**Status**: Ready to begin Week 10 (Async Metal Execution)! 🚀

**Created**: December 9, 2024  
**Phase**: 5 of 6  
**Progress**: 9/14 weeks complete (64%)

