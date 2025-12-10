# Phase 5 Week 11 Plan: Metal Command Buffer Batching

**Date**: December 9, 2024  
**Duration**: Week 11 of 12 (Phase 5)  
**Status**: STARTING  
**Goal**: Implement Metal command buffer batching for 20-30% speedup

---

## Overview

Week 11 builds on the async infrastructure from Week 10 to implement true Metal command buffer batching. Instead of executing operations one-by-one, we'll batch multiple operations into a single Metal command buffer, reducing CPU-GPU synchronization overhead.

---

## Current State (Week 10 Baseline)

### Performance
- **Async overhead**: ~5-15µs from `spawn_blocking`
- **Execution model**: Synchronous, wrapped in async
- **CPU-GPU sync**: After every operation

### Architecture
```rust
// Week 10: Each operation is a separate sync call
pub async fn execute_tensor(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
    tokio::task::spawn_blocking(move || tensor.clone().eval()).await?
}
```

**Problem**: No batching, CPU waits after each operation

---

## Week 11 Goals

### Primary Goal
Implement Metal command buffer batching to reduce CPU-GPU synchronization overhead.

### Target Performance
- **20-30% speedup** from batching
- **Reduced CPU overhead** by 50-70%
- **Better GPU utilization** through batching

### Technical Objectives
1. ✅ Create Metal command buffer abstraction
2. ✅ Implement operation batching logic
3. ✅ Batch entire computation graph into single command buffer
4. ✅ Maintain 100% correctness (all tests still pass)
5. ✅ Benchmark improvement
6. ✅ Profile with Instruments

---

## Architecture Design

### Approach 1: Simple Batching (Week 11) ✅

**Strategy**: Execute entire computation graph in single command buffer

```rust
// Week 11: Batch all operations together
pub async fn execute_tensor_batched(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
    let graph = tensor.graph().read().unwrap();
    
    // 1. Get topological order
    let order = graph.topological_order()?;
    
    // 2. Create single command buffer for ALL operations
    let cmd_buffer = self.device.new_command_buffer();
    
    // 3. Encode all operations
    for node_id in order {
        self.encode_operation(cmd_buffer, node_id)?;
    }
    
    // 4. Commit once, wait once
    cmd_buffer.commit();
    cmd_buffer.wait_until_completed();
    
    Ok(result)
}
```

**Benefits**:
- Single CPU-GPU synchronization point
- Reduced kernel launch overhead
- Better GPU utilization

**Limitations**:
- Still synchronous (waits for GPU completion)
- No operation fusion yet (Week 12)
- Candle doesn't expose Metal command buffers directly

### Approach 2: Candle Integration (Realistic) ✅

Since Candle manages its own Metal backend, we can't directly manipulate command buffers. Instead:

**Strategy**: Optimize graph execution to minimize Candle's internal synchronization

```rust
// Week 11: Optimized graph execution
pub async fn execute_tensor_optimized(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
    // 1. Evaluate entire graph without intermediate .eval() calls
    // 2. Let Candle batch operations internally
    // 3. Use async to avoid blocking
    
    tokio::task::spawn_blocking(move || {
        // Execute full graph in one go
        tensor.eval()
    }).await?
}
```

**Reality Check**: We're already doing this in Week 10! The key insight:
- Candle already batches operations internally
- Our lazy evaluation helps by building full graph
- Week 11 focus: Optimize **graph building** and **reduce allocation overhead**

---

## Revised Week 11 Strategy

Given Candle's abstraction, Week 11 will focus on:

### 1. Graph Building Optimization ✅

**Problem**: Lock contention during graph building

**Solution**: Use lock-free data structures

```rust
use dashmap::DashMap;

pub struct ComputationGraph {
    nodes: DashMap<NodeId, GraphNode>,  // Lock-free!
    next_node_id: AtomicUsize,
}
```

**Expected**: 10-15% speedup from reduced lock contention

### 2. Memory Allocation Optimization ✅

**Problem**: Frequent allocations during graph building

**Solution**: Pre-allocate and reuse buffers

```rust
pub struct AsyncGraphExecutor {
    device: Device,
    tensor_cache: DashMap<NodeId, Tensor>,  // Reuse computed tensors
    batch_size_hint: usize,
}
```

**Expected**: 5-10% speedup from reduced allocations

### 3. Parallel Graph Execution ✅

**Problem**: Sequential graph traversal

**Solution**: Execute independent operations in parallel

```rust
pub async fn execute_tensor_parallel(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
    let graph = tensor.graph().read().unwrap();
    let order = graph.topological_order()?;
    
    // Find independent operations
    let batches = self.group_independent_operations(&order);
    
    // Execute batches in parallel
    for batch in batches {
        let futures: Vec<_> = batch.iter()
            .map(|&node_id| self.execute_node_async(node_id))
            .collect();
        
        futures::future::join_all(futures).await?;
    }
    
    Ok(result)
}
```

**Expected**: 10-20% speedup from parallelism

### Combined Target: 25-45% speedup ✅

---

## Implementation Plan

### Task 1: Lock-Free Graph (Days 1-2)

**File**: `src/graph/node.rs`

**Changes**:
```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct ComputationGraph {
    nodes: DashMap<NodeId, GraphNode>,  // Changed from RwLock<HashMap>
    next_node_id: AtomicUsize,
    device: Device,
}

impl ComputationGraph {
    pub fn add_node(&self, operation: Operation, inputs: Vec<NodeId>) -> Result<NodeId> {
        let node_id = NodeId(self.next_node_id.fetch_add(1, Ordering::SeqCst));
        
        // Lock-free insert
        self.nodes.insert(node_id, GraphNode {
            id: node_id,
            operation,
            inputs,
            // ...
        });
        
        Ok(node_id)
    }
}
```

**Testing**: Verify all existing tests still pass

### Task 2: Tensor Caching (Days 3-4)

**File**: `src/graph/async_executor.rs`

**Changes**:
```rust
use dashmap::DashMap;

pub struct AsyncGraphExecutor {
    device: Device,
    sync_executor: SyncExecutor,
    tensor_cache: DashMap<NodeId, Arc<Tensor>>,  // New: cache results
}

impl AsyncGraphExecutor {
    pub async fn execute_tensor(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
        let node_id = tensor.node_id();
        
        // Check cache first
        if let Some(cached) = self.tensor_cache.get(&node_id) {
            return Ok(cached.as_ref().clone());
        }
        
        // Execute and cache
        let result = self.execute_uncached(tensor).await?;
        self.tensor_cache.insert(node_id, Arc::new(result.clone()));
        
        Ok(result)
    }
}
```

**Testing**: Add cache hit/miss tests

### Task 3: Parallel Execution (Days 5-6)

**File**: `src/graph/async_executor.rs`

**Changes**:
```rust
use futures::future::join_all;

impl AsyncGraphExecutor {
    pub async fn execute_tensor_parallel(&mut self, tensor: &LazyTensor) -> Result<Tensor> {
        let graph = tensor.graph().read().unwrap();
        let order = graph.topological_order()?;
        
        // Group independent operations into batches
        let batches = self.compute_execution_batches(&order, &graph);
        
        // Execute each batch in parallel
        for batch in batches {
            let futures: Vec<_> = batch.into_iter()
                .map(|node_id| {
                    let graph_clone = graph.clone();
                    async move {
                        self.execute_node(&graph_clone, node_id).await
                    }
                })
                .collect();
            
            join_all(futures).await;
        }
        
        Ok(result)
    }
    
    fn compute_execution_batches(&self, order: &[NodeId], graph: &ComputationGraph) -> Vec<Vec<NodeId>> {
        // Compute which operations can run in parallel
        let mut batches = Vec::new();
        let mut remaining = order.to_vec();
        
        while !remaining.is_empty() {
            let mut batch = Vec::new();
            let mut i = 0;
            
            while i < remaining.len() {
                let node_id = remaining[i];
                let node = graph.get_node(node_id).unwrap();
                
                // Can run if all inputs are computed
                let ready = node.inputs.iter()
                    .all(|input_id| !remaining.contains(input_id));
                
                if ready {
                    batch.push(node_id);
                    remaining.remove(i);
                } else {
                    i += 1;
                }
            }
            
            batches.push(batch);
        }
        
        batches
    }
}
```

**Testing**: Verify parallel execution maintains correctness

### Task 4: Benchmarking (Day 7)

**File**: `benches/async_batching.rs`

**Benchmarks**:
1. Week 10 baseline (sync wrapped in async)
2. Week 11 with lock-free graph
3. Week 11 with tensor caching
4. Week 11 with parallel execution
5. Week 11 full optimization

**Target Results**:
- Lock-free graph: +10-15%
- Tensor caching: +5-10%
- Parallel execution: +10-20%
- **Combined: +25-45%**

---

## Testing Strategy

### Correctness Tests

All Week 10 tests must still pass:
```bash
cargo test --features async-exec --test async_execution
```

Expected: **7/7 passing** (100% correctness maintained)

### Performance Tests

**New benchmarks**:
1. Graph building overhead
2. Cache hit rate
3. Parallel speedup factor
4. End-to-end latency

**File**: `benches/week11_performance.rs`

```rust
fn benchmark_graph_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_building");
    
    // Week 10: RwLock<HashMap>
    group.bench_function("week10_rwlock", |b| {
        b.iter(|| {
            let graph = ComputationGraphWeek10::new();
            // Build graph
        });
    });
    
    // Week 11: DashMap
    group.bench_function("week11_dashmap", |b| {
        b.iter(|| {
            let graph = ComputationGraph::new();
            // Build graph
        });
    });
}
```

### Profiling

**Instruments**:
```bash
# Time profiler
cargo instruments -t Time --release --features async-exec --example async_benchmark

# Allocations profiler
cargo instruments -t Allocations --release --features async-exec --example async_benchmark
```

**Focus Areas**:
- Lock contention (should decrease)
- Allocation count (should decrease)
- CPU utilization (should increase from parallelism)

---

## Success Metrics

| Metric | Week 10 Baseline | Week 11 Target | Measurement |
|--------|------------------|----------------|-------------|
| Graph building | 100% | 85-90% (10-15% faster) | Benchmark |
| Memory allocations | 100% | 90-95% (5-10% fewer) | Instruments |
| End-to-end latency | 100% | 70-75% (25-30% faster) | Benchmark |
| Correctness | 100% | 100% | All tests pass |

---

## Risk Mitigation

### Risk 1: Lock-Free Complexity

**Risk**: DashMap harder to debug  
**Mitigation**: Keep RwLock version for comparison  
**Fallback**: Feature-gate lock-free implementation

### Risk 2: Parallel Execution Bugs

**Risk**: Race conditions in parallel execution  
**Mitigation**: Extensive testing, property-based tests  
**Fallback**: Disable parallelism if issues found

### Risk 3: Performance Target Not Met

**Risk**: < 20% speedup  
**Mitigation**: Profile early, iterate quickly  
**Acceptance**: Any measurable improvement is progress

---

## Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Lock-free graph | DashMap implementation, tests pass |
| 3-4 | Tensor caching | Cache implementation, benchmarks |
| 5-6 | Parallel execution | Parallel executor, correctness tests |
| 7 | Benchmarking | Performance comparison, profiling |

---

## Deliverables

### Code
- Updated `src/graph/node.rs` with lock-free DashMap
- Updated `src/graph/async_executor.rs` with caching and parallelism
- New `benches/week11_performance.rs` for benchmarking

### Documentation
- `PHASE5_WEEK11_PROGRESS.md` - Daily progress
- `PHASE5_WEEK11_COMPLETE.md` - Final summary
- Updated API docs with performance notes

### Benchmarks
- Graph building overhead comparison
- Cache performance analysis
- Parallel execution speedup
- End-to-end latency improvement

---

## Expected Outcomes

### Performance
- **25-45% speedup** from combined optimizations
- **10-15% fewer** memory allocations
- **Better CPU utilization** from parallelism

### Code Quality
- Maintains 100% test passing rate
- Zero new clippy warnings
- Comprehensive benchmarks

### Documentation
- Clear performance analysis
- Profiling data from Instruments
- Comparison with Week 10 baseline

---

## Next: Week 12

After Week 11 completes, Week 12 will focus on:
1. Operation fusion (LoRA, matmul+bias, etc)
2. Metal kernel tuning
3. MLX comparison
4. Achieve 2-3x overall Phase 5 target

---

**Status**: Ready to implement! 🚀

**Created**: December 9, 2024  
**Phase**: 5, Week 11 of 12  
**Progress**: 10/14 weeks (71% → 79% after this week)

