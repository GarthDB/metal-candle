# metal-candle v2.0 Architecture Design

**Date**: December 9, 2024  
**Based On**: MLX Architecture Analysis  
**Goal**: Achieve 80-100% of MLX performance through lazy evaluation + async execution

---

## Design Philosophy

### Core Principles

1. **Lazy by Default, Explicit Eval** - Operations build a graph, execution is explicit
2. **Async GPU Execution** - CPU and GPU work overlap via Metal command buffers
3. **Zero-Copy When Possible** - Minimize memory allocations and transfers
4. **Backward Compatible** - v1.0 API continues to work (with performance hit)
5. **Production Quality** - Same high standards as v1.0 (tests, docs, error handling)

### Performance Targets

| Operation | v1.0 (Eager) | v2.0 Target | MLX (Goal) |
|-----------|--------------|-------------|------------|
| LoRA (3 ops) | 36 µs | 10-15 µs (2-3x) | 5-11 µs |
| Softmax | 39 µs | 5-8 µs (5-8x) | 1.85 µs |
| RMS Norm | 47 µs | 10-15 µs (3-5x) | 6 µs |
| Training Step | ~500 µs | ~200 µs (2.5x) | ~150 µs |

**Expected Gains**:
- Async execution: 40-50% reduction
- Command buffer batching: 20-30% reduction
- Improved kernels: 10-20% reduction

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User-Facing API                         │
│  LazyTensor (new) + Tensor (v1.0 compat wrapper)           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                  Computation Graph                           │
│  - Graph nodes (operations + inputs)                        │
│  - Topological ordering                                     │
│  - Optimization passes (Phase 4+)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                   Async Executor                             │
│  - Command buffer management                                │
│  - Batched operation encoding                               │
│  - Async completion tracking                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│                   Primitives Layer                           │
│  - MatmulPrimitive, LoRAPrimitive, SoftmaxPrimitive        │
│  - Wraps Candle ops + custom Metal kernels                 │
└─────────────────────────────────────────────────────────────┘
                         │
                         v
┌─────────────────────────────────────────────────────────────┐
│               Metal Backend (Candle + Custom)               │
│  - Candle's Metal backend for standard ops                 │
│  - Our custom kernels (LoRA, Softmax, RMS Norm)           │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. LazyTensor: The Core Abstraction

### Design

```rust
/// A tensor that defers computation until `.eval()` is called.
///
/// LazyTensors are lightweight graph nodes that record operations
/// without executing them. Multiple operations can be chained and
/// executed together in a single command buffer for efficiency.
///
/// # Examples
///
/// ```rust
/// use metal_candle::LazyTensor;
///
/// // Create lazy tensors (no computation yet)
/// let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
/// let b = LazyTensor::from_slice(&[4.0, 5.0, 6.0], &[3])?;
///
/// // Build computation graph (still no execution)
/// let c = a.add(&b)?;
/// let d = c.mul(&2.0)?;
///
/// // Execute entire graph at once
/// let result = d.eval()?;  // Now computation happens
/// ```
pub struct LazyTensor {
    /// Unique node ID in the computation graph
    node_id: NodeId,
    
    /// Shared computation graph (Arc for cheap cloning)
    graph: Arc<RwLock<ComputationGraph>>,
    
    /// Output shape (known without evaluation)
    shape: Shape,
    
    /// Output dtype (known without evaluation)
    dtype: DType,
    
    /// Device this tensor will execute on
    device: Device,
}

/// Unique identifier for a graph node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);
```

### Key Methods

```rust
impl LazyTensor {
    /// Create a lazy tensor from data (graph leaf node)
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self>;
    
    /// Operations - return new LazyTensors without executing
    pub fn matmul(&self, other: &LazyTensor) -> Result<LazyTensor>;
    pub fn add(&self, other: &LazyTensor) -> Result<LazyTensor>;
    pub fn mul(&self, scalar: f32) -> Result<LazyTensor>;
    pub fn softmax(&self, dim: usize) -> Result<LazyTensor>;
    pub fn rms_norm(&self, eps: f32) -> Result<LazyTensor>;
    
    /// Execute the computation graph and return a concrete Tensor
    pub fn eval(self) -> Result<Tensor>;
    
    /// Get shape/dtype without evaluation
    pub fn shape(&self) -> &[usize];
    pub fn dtype(&self) -> DType;
    pub fn device(&self) -> &Device;
}
```

### Memory Management

```rust
/// Graph nodes don't own data until evaluated
pub enum NodeData {
    /// Input data (owned)
    Concrete(Tensor),
    
    /// Not yet computed
    Lazy,
    
    /// Being computed (Metal command buffer in flight)
    Evaluating {
        buffer: Arc<metal::Buffer>,
        event: metal::Event,
    },
    
    /// Computation complete
    Available(Tensor),
}
```

---

## 2. Computation Graph

### Design

```rust
/// A computation graph representing deferred operations.
///
/// The graph is a DAG (Directed Acyclic Graph) where nodes are operations
/// and edges are dependencies. Topological execution order ensures all
/// inputs are ready before an operation executes.
pub struct ComputationGraph {
    /// All nodes in the graph
    nodes: Vec<GraphNode>,
    
    /// Device for execution
    device: Device,
    
    /// Metal command buffer (shared across operations)
    command_buffer: Option<Arc<metal::CommandBuffer>>,
    
    /// Pending operations (not yet encoded)
    pending_ops: Vec<NodeId>,
}

pub struct GraphNode {
    /// Node ID
    id: NodeId,
    
    /// The operation
    operation: Operation,
    
    /// Input node IDs
    inputs: Vec<NodeId>,
    
    /// Output shape and dtype
    output_shape: Shape,
    output_dtype: DType,
    
    /// Actual data (if evaluated)
    data: NodeData,
}

/// All supported operations
pub enum Operation {
    /// Input data (leaf node)
    Input,
    
    /// Matrix multiplication
    Matmul,
    
    /// Element-wise addition
    Add,
    
    /// Scalar multiplication
    MulScalar { value: f32 },
    
    /// Fused LoRA operation (input @ A @ B * scale)
    LoRA { a: NodeId, b: NodeId, scale: f32 },
    
    /// Fused softmax
    Softmax { dim: usize },
    
    /// Fused RMS normalization
    RMSNorm { eps: f32 },
    
    // ... more operations
}
```

### Key Methods

```rust
impl ComputationGraph {
    /// Add a new operation node to the graph
    pub fn add_node(
        &mut self,
        operation: Operation,
        inputs: Vec<NodeId>,
        output_shape: Shape,
        output_dtype: DType,
    ) -> NodeId;
    
    /// Execute the graph starting from a specific node
    pub fn execute(&mut self, output_node: NodeId) -> Result<Tensor>;
    
    /// Get topological execution order
    fn topological_order(&self, output_node: NodeId) -> Vec<NodeId>;
    
    /// Optimize the graph (Phase 4+)
    pub fn optimize(&mut self);
}
```

### Topological Execution

```rust
impl ComputationGraph {
    fn topological_order(&self, output_node: NodeId) -> Vec<NodeId> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        
        fn visit(
            graph: &ComputationGraph,
            node_id: NodeId,
            visited: &mut HashSet<NodeId>,
            order: &mut Vec<NodeId>,
        ) {
            if visited.contains(&node_id) {
                return;
            }
            visited.insert(node_id);
            
            // Visit dependencies first (post-order)
            for &input_id in &graph.nodes[node_id.0].inputs {
                visit(graph, input_id, visited, order);
            }
            
            order.push(node_id);
        }
        
        visit(self, output_node, &mut visited, &mut order);
        order
    }
}
```

---

## 3. Async Executor

### Design

```rust
/// Executes computation graphs asynchronously using Metal.
///
/// The executor batches multiple operations into a single command buffer
/// to minimize Metal overhead. GPU work executes asynchronously while
/// the CPU continues.
pub struct AsyncExecutor {
    /// Metal device
    device: metal::Device,
    
    /// Command queue
    command_queue: Arc<metal::CommandQueue>,
    
    /// Current command buffer (shared across operations)
    current_buffer: Option<CommandBufferState>,
    
    /// Maximum operations per command buffer before commit
    max_ops_per_buffer: usize,
}

struct CommandBufferState {
    buffer: Arc<metal::CommandBuffer>,
    encoder: metal::ComputeCommandEncoder,
    num_encoded_ops: usize,
}
```

### Key Methods

```rust
impl AsyncExecutor {
    /// Execute a single operation, encoding into the current command buffer
    pub fn execute_operation(
        &mut self,
        operation: &Operation,
        inputs: &[Tensor],
    ) -> Result<Tensor>;
    
    /// Commit the current command buffer (async)
    pub fn commit(&mut self) -> Result<metal::Event>;
    
    /// Wait for all pending operations to complete
    pub fn synchronize(&mut self) -> Result<()>;
    
    /// Get or create a command buffer
    fn get_or_create_command_buffer(&mut self) -> Result<&mut CommandBufferState>;
}
```

### Async Execution Flow

```rust
impl AsyncExecutor {
    pub fn execute_operation(
        &mut self,
        operation: &Operation,
        inputs: &[Tensor],
    ) -> Result<Tensor> {
        // Get shared command buffer
        let cb_state = self.get_or_create_command_buffer()?;
        
        // Encode operation into command buffer
        match operation {
            Operation::Matmul => {
                encode_matmul(&mut cb_state.encoder, &inputs[0], &inputs[1])?;
            }
            Operation::LoRA { a, b, scale } => {
                // Use our custom fused kernel
                encode_lora(&mut cb_state.encoder, inputs, a, b, *scale)?;
            }
            // ... other operations
        }
        
        cb_state.num_encoded_ops += 1;
        
        // Commit if buffer is full
        if cb_state.num_encoded_ops >= self.max_ops_per_buffer {
            self.commit()?;
        }
        
        // Return output tensor (data not yet available!)
        Ok(Tensor::lazy_output(...))
    }
    
    pub fn commit(&mut self) -> Result<metal::Event> {
        let cb_state = self.current_buffer.take().unwrap();
        
        // End encoding
        cb_state.encoder.end_encoding();
        
        // Commit command buffer (ASYNC!)
        cb_state.buffer.commit();
        
        // Return event for synchronization
        Ok(cb_state.buffer.event())
    }
}
```

---

## 4. Primitives Layer

### Design

```rust
/// Abstract interface for ML operations.
///
/// Primitives provide a platform-agnostic interface similar to MLX.
/// Each primitive can have multiple backend implementations.
pub trait Primitive {
    /// Execute on Metal (using Candle or custom kernels)
    fn eval_metal(
        &self,
        encoder: &mut metal::ComputeCommandEncoder,
        inputs: &[&metal::Buffer],
        output: &metal::Buffer,
    ) -> Result<()>;
    
    /// Execute on CPU (fallback)
    fn eval_cpu(&self, inputs: &[&Tensor]) -> Result<Tensor>;
    
    /// Operation name for debugging
    fn name(&self) -> &'static str;
    
    /// Compute output shape
    fn output_shape(&self, input_shapes: &[&[usize]]) -> Result<Vec<usize>>;
}
```

### Example: LoRA Primitive

```rust
pub struct LoRAPrimitive {
    lora_a: Tensor,
    lora_b: Tensor,
    scale: f32,
}

impl Primitive for LoRAPrimitive {
    fn eval_metal(
        &self,
        encoder: &mut metal::ComputeCommandEncoder,
        inputs: &[&metal::Buffer],
        output: &metal::Buffer,
    ) -> Result<()> {
        // Use our existing FusedLoRAOp kernel
        let kernel = get_lora_kernel()?;
        encoder.set_compute_pipeline_state(&kernel);
        
        // Set buffers
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(&self.lora_a.buffer()), 0);
        encoder.set_buffer(2, Some(&self.lora_b.buffer()), 0);
        encoder.set_buffer(3, Some(output), 0);
        
        // Dispatch
        let grid_size = compute_grid_size(...);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        
        Ok(())
    }
    
    fn name(&self) -> &'static str {
        "lora"
    }
}
```

---

## 5. Backward Compatibility Layer

### Design

Keep v1.0 API working with minimal changes:

```rust
// v1.0 Tensor (eager evaluation)
impl Tensor {
    /// Matrix multiplication (eager, for backward compatibility)
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Convert to LazyTensor, execute, return immediately
        LazyTensor::from_tensor(self.clone())?
            .matmul(&LazyTensor::from_tensor(other.clone())?)?
            .eval()
    }
    
    // Similar for other operations
}
```

**Performance Impact**: v1.0 API will be slower than v2.0 due to immediate `.eval()` calls, but still functional.

### Migration Path

```rust
// v1.0 (eager) - still works, but slower
let output = input.matmul(&weight)?;

// v2.0 (lazy) - faster, recommended
let output = input.to_lazy()?
    .matmul(&weight.to_lazy()?)?
    .eval()?;

// Or use new API directly
let output = LazyTensor::from_tensor(input)?
    .matmul(&LazyTensor::from_tensor(weight)?)?
    .eval()?;
```

---

## 6. Error Handling

### Graph-Specific Errors

```rust
#[derive(Error, Debug)]
pub enum GraphError {
    #[error("node {id:?} not found in graph")]
    NodeNotFound { id: NodeId },
    
    #[error("circular dependency detected involving node {id:?}")]
    CircularDependency { id: NodeId },
    
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    
    #[error("operation {operation} failed: {reason}")]
    OperationFailed {
        operation: String,
        reason: String,
    },
}
```

### Async Execution Errors

```rust
#[derive(Error, Debug)]
pub enum ExecutorError {
    #[error("command buffer execution failed: {reason}")]
    CommandBufferFailed { reason: String },
    
    #[error("Metal device error: {0}")]
    MetalError(#[from] metal::DeviceError),
    
    #[error("timeout waiting for GPU completion")]
    Timeout,
}
```

---

## 7. Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_lazy_tensor_graph_building() {
        // Verify operations don't execute immediately
        let a = LazyTensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let b = LazyTensor::from_slice(&[3.0, 4.0], &[2]).unwrap();
        
        let c = a.add(&b).unwrap();
        
        // Graph should have 3 nodes, no execution yet
        assert_eq!(c.graph_size(), 3);
        assert!(!c.is_evaluated());
    }
    
    #[test]
    fn test_lazy_eval_correctness() {
        let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b = LazyTensor::from_slice(&[4.0, 5.0, 6.0], &[3]).unwrap();
        
        let c = a.add(&b).unwrap().eval().unwrap();
        
        assert_eq!(c.to_vec::<f32>(), vec![5.0, 7.0, 9.0]);
    }
}
```

### Integration Tests

```rust
#[test]
fn test_lora_forward_lazy_vs_eager() {
    // Compare lazy execution vs eager
    let input = Tensor::randn(&[128, 512], DType::F32, &device)?;
    let lora_a = Tensor::randn(&[512, 8], DType::F32, &device)?;
    let lora_b = Tensor::randn(&[8, 512], DType::F32, &device)?;
    
    // Eager (v1.0)
    let eager_start = Instant::now();
    let eager_output = input.matmul(&lora_a)?
        .matmul(&lora_b)?
        .mul(0.1)?;
    let eager_time = eager_start.elapsed();
    
    // Lazy (v2.0)
    let lazy_start = Instant::now();
    let lazy_output = input.to_lazy()?
        .matmul(&lora_a.to_lazy()?)?
        .matmul(&lora_b.to_lazy()?)?
        .mul(0.1)?
        .eval()?;
    let lazy_time = lazy_start.elapsed();
    
    // Verify correctness
    assert_tensors_close(&eager_output, &lazy_output, 1e-5);
    
    // Verify performance
    assert!(lazy_time < eager_time * 2 / 3);  // At least 1.5x faster
}
```

### Benchmark Suite

```rust
// benches/lazy_vs_eager.rs
fn benchmark_lora_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_forward");
    
    group.bench_function("eager", |b| {
        b.iter(|| {
            let output = input.matmul(&lora_a)
                .unwrap()
                .matmul(&lora_b)
                .unwrap()
                .mul(scale)
                .unwrap();
            black_box(output);
        });
    });
    
    group.bench_function("lazy", |b| {
        b.iter(|| {
            let output = input_lazy.matmul(&lora_a_lazy)
                .unwrap()
                .matmul(&lora_b_lazy)
                .unwrap()
                .mul(scale)
                .unwrap()
                .eval()
                .unwrap();
            black_box(output);
        });
    });
    
    group.finish();
}
```

---

## 8. Implementation Timeline

### Phase 1: Core Infrastructure (Weeks 4-6)

**Week 4**:
- [ ] Implement `LazyTensor` struct and basic methods
- [ ] Implement `ComputationGraph` with node management
- [ ] Write unit tests for graph building

**Week 5**:
- [ ] Implement `AsyncExecutor` with command buffer management
- [ ] Add topological execution order
- [ ] Integrate with existing Metal kernels

**Week 6**:
- [ ] Implement `Primitive` trait and adapters for Candle ops
- [ ] Add error handling and validation
- [ ] Write integration tests

### Phase 2: Operation Migration (Weeks 7-9)

**Week 7**:
- [ ] Migrate LoRALayer to use `LazyTensor`
- [ ] Add `LoRAPrimitive` using existing fused kernel
- [ ] Benchmark and validate

**Week 8**:
- [ ] Migrate Softmax and RMS Norm
- [ ] Add corresponding primitives
- [ ] Update tests

**Week 9**:
- [ ] Migrate remaining operations (matmul, add, mul, etc.)
- [ ] Implement backward compatibility layer
- [ ] Comprehensive testing

### Phase 3: Optimization (Weeks 10-12)

**Week 10**:
- [ ] Profile with Instruments
- [ ] Optimize graph overhead
- [ ] Tune command buffer batching

**Week 11**:
- [ ] Add basic optimization passes (optional)
- [ ] Optimize memory allocation
- [ ] Reduce synchronization points

**Week 12**:
- [ ] Final benchmarking vs MLX
- [ ] Performance tuning
- [ ] Documentation updates

---

## 9. Success Metrics

### Performance

- [ ] LoRA forward: < 15 µs (target: 10-15 µs, MLX: 5-11 µs)
- [ ] Softmax: < 8 µs (target: 5-8 µs, MLX: 1.85 µs)
- [ ] RMS Norm: < 15 µs (target: 10-15 µs, MLX: 6 µs)
- [ ] Training step: < 250 µs (target: ~200 µs, MLX: ~150 µs)

### Correctness

- [ ] All v1.0 tests pass with backward compat layer
- [ ] New lazy execution tests pass
- [ ] Numerical correctness validated (max error < 1e-5)

### Code Quality

- [ ] Clippy pedantic passes with zero warnings
- [ ] Coverage ≥ 80% (same as v1.0)
- [ ] All public APIs documented
- [ ] Migration guide complete

---

## 10. Open Questions

1. **Graph Size Limits**: What's the max nodes before performance degrades?
   - **Answer**: Test with 100-1000 node graphs in benchmarks

2. **Memory Pressure**: When to release intermediate buffers?
   - **Answer**: Release after eval, use weak refs for shared buffers

3. **Multi-Stream Support**: Do we need it for v2.0?
   - **Answer**: No, single stream sufficient initially

4. **Optimization Passes**: How complex should they be?
   - **Answer**: Start simple (pattern matching), defer JIT to v2.1+

5. **Async API**: Should we use `async/await`?
   - **Answer**: No, complexity not worth it. Use Events for sync.

---

## Conclusion

This design balances:
- **Performance**: Targeting 80-100% of MLX through async execution
- **Simplicity**: No JIT or complex graph optimization initially
- **Compatibility**: v1.0 API continues to work
- **Quality**: Same high standards as v1.0

The architecture is heavily inspired by MLX but adapted for Rust and Candle integration. By focusing on lazy evaluation and async command buffers (the main source of MLX's advantage), we can achieve most of the performance gains without the complexity of full graph optimization.

Next steps: Begin implementation in Phase 3 (Weeks 4-6).

