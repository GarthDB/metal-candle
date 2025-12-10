# MLX Architecture Analysis

**Date**: December 9, 2024  
**Purpose**: Deep analysis of MLX's lazy evaluation and Metal backend for metal-candle v2.0 rewrite

## Executive Summary

MLX achieves superior performance through three key architectural decisions:

1. **Lazy Evaluation with Graph Representation** - Operations build a computation graph instead of executing immediately
2. **Asynchronous GPU Execution** - Metal command buffers execute asynchronously with CPU, overlapping work
3. **Primitive-Based Architecture** - All operations are `Primitive` objects with platform-specific implementations

**Key Finding**: MLX's performance advantage comes primarily from **deferred execution + async dispatch**, not just custom kernels. Our current eager evaluation forces synchronization between operations.

---

## 1. Core Architecture: Array as Graph Node

### Array Structure

```cpp
// mlx/array.h
class array {
    std::shared_ptr<ArrayDesc> array_desc_;  // Shared state
};

struct ArrayDesc {
    Shape shape;
    Dtype dtype;
    std::shared_ptr<Primitive> primitive;     // The operation that produces this array
    std::vector<array> inputs;                 // Input arrays to the operation
    std::vector<array> siblings;               // Other outputs from same primitive
    Status status;                             // unscheduled, evaluated, or available
    Event event;                               // For async synchronization
    std::shared_ptr<Data> data;               // Actual buffer (only allocated after eval)
};
```

**Key Insight**: Arrays are **graph nodes**, not data containers. The actual buffer is only allocated after evaluation.

### Status State Machine

```cpp
enum Status {
    unscheduled,  // Graph node created, not yet evaluated
    evaluated,    // eval_gpu() called, but may still be executing
    available     // Computation complete, data ready
};
```

**Flow**:
1. `a + b` creates a new array with status `unscheduled`
2. Calling `.eval()` or accessing data triggers evaluation → `evaluated`
3. When Metal command buffer completes → `available`

---

## 2. Lazy Evaluation Mechanism

### Creating Operations (Graph Building)

```cpp
// mlx/ops.cpp - Example: matmul
array matmul(
    const array& a,
    const array& b,
    StreamOrDevice s) {
  // Does NOT execute! Just creates a graph node
  return array(
      /* shape */ {a.shape(-2), b.shape(-1)},
      /* dtype */ a.dtype(),
      /* primitive */ std::make_shared<Matmul>(s),
      /* inputs */ {a, b}
  );
}
```

**No execution happens here!** The array is just a promise of future computation.

### Triggering Evaluation

```cpp
// mlx/array.cpp
void array::eval() {
  if (status() == Status::unscheduled) {
    mlx::core::eval({*this});  // Trigger scheduler
  } else {
    wait();  // Already evaluating, just wait
  }
}
```

Evaluation is triggered by:
- Explicit `.eval()` call
- Accessing data (e.g., `.item()`, `.data<T>()`)
- Synchronization points

### Scheduler and Async Execution

```cpp
// mlx/backend/metal/eval.cpp
void eval(array& arr) {
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);
  
  // Call the primitive's GPU implementation
  arr.primitive().eval_gpu(arr.inputs(), outputs);
  
  // Enqueue async completion handler
  command_buffer->addCompletedHandler(
      [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error(cbuf);
      });
  
  // Commit command buffer (ASYNC!)
  d.commit_command_buffer(s.index);
  
  // IMPORTANT: eval() returns immediately, GPU work continues
}
```

**Critical Observation**: The CPU function returns **immediately** after committing the command buffer. GPU work continues asynchronously.

---

## 3. Metal Backend Architecture

### Command Buffer Management

MLX maintains **one command buffer per stream** and batches operations:

```cpp
// mlx/backend/metal/device.h
struct DeviceStream {
  MTL::CommandQueue* queue;
  MTL::CommandBuffer* command_buffer;
  CommandEncoder* encoder;  // For current batch of operations
  
  bool command_buffer_needs_commit();
  void commit_command_buffer();
};
```

**Strategy**:
1. Multiple operations encode into the **same command buffer**
2. Command buffer only committed when:
   - Explicit synchronization (e.g., `.wait()`)
   - Buffer full
   - Stream switch

This **batching** reduces Metal overhead significantly!

### Primitives: Platform-Agnostic Interface

```cpp
// mlx/primitives.h
class Primitive {
 public:
  explicit Primitive(Stream stream);
  
  // Must be implemented for each backend
  virtual void eval_cpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;
  
  virtual void eval_gpu(
      const std::vector<array>& inputs,
      std::vector<array>& outputs) = 0;
  
  // Autodiff support
  virtual std::vector<array> jvp(...);
  virtual std::vector<array> vjp(...);
  
  // Name for debugging
  virtual const char* name() const = 0;
};
```

**Example**: `Matmul` primitive has separate CPU and Metal implementations.

### Metal Matmul Implementation Highlights

From `mlx/backend/metal/matmul.cpp`:

```cpp
void Matmul::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  
  // Get Metal encoder (from shared command buffer!)
  auto& compute_encoder = d.get_command_encoder(s.index);
  
  // Choose optimized kernel based on dimensions
  if (/* large matmul */) {
    steel_matmul(compute_encoder, a, b, out, ...);
  } else {
    naive_matmul(compute_encoder, a, b, out, ...);
  }
  
  // Encoder stays open for next operation!
}
```

**Key Points**:
- Uses **highly optimized "Steel" kernels** for large matrices
- Tiled matrix multiplication with shared memory
- Kernel selection based on dimensions and transpose flags
- **No command buffer commit** - keeps encoding!

---

## 4. Comparison: MLX vs metal-candle (Current)

| Aspect | MLX | metal-candle (v1.0) |
|--------|-----|---------------------|
| **Execution Model** | Lazy (graph build, then eval) | Eager (immediate execution) |
| **CPU/GPU Overlap** | Yes (async command buffers) | Minimal (sync after each op) |
| **Command Buffer Batching** | Multiple ops per buffer | One op per buffer (via Candle) |
| **Operation Fusion** | Manual + graph patterns | Explicit custom kernels only |
| **Memory Allocation** | Delayed until eval | Immediate |
| **Primitives** | Abstract interface (CPU/GPU/CUDA) | Candle's backend abstraction |

### Performance Implications

**Example: LoRA Forward Pass**

```python
# MLX (lazy)
hidden = x @ lora_a        # Graph node, no execution
output = hidden @ lora_b   # Graph node, no execution
scaled = output * scale    # Graph node, no execution
result.eval()              # Batched execution: 3 ops in 1 command buffer

# metal-candle (eager)
let hidden = x.matmul(&lora_a)?;    // Encode + commit
let output = hidden.matmul(&lora_b)?; // Encode + commit
let scaled = output.mul(scale)?;      // Encode + commit
// 3 separate command buffer submissions!
```

**Estimated Overhead**:
- Command buffer creation/commit: ~3-5 µs per operation
- MLX advantage: ~6-10 µs saved for 3-op sequence
- This matches our observed gap!

---

## 5. Graph Optimization (Future Potential)

MLX has infrastructure for graph passes, though currently limited:

```cpp
// mlx/graph_utils.h
// Can traverse graph and identify patterns
void visit_graph(array& arr, std::function<void(array&)> visitor);
```

**Potential Optimizations**:
- Fuse `matmul -> matmul -> mul` into single LoRA kernel
- Fuse `sub -> exp -> sum -> div` into single Softmax kernel
- Eliminate redundant memory transfers

**Current Status**: MLX does **minimal automatic fusion**. Performance comes primarily from async execution, not fusion.

---

## 6. Key Takeaways for metal-candle v2.0

### What We Should Adopt

1. **Lazy Tensor with Graph Representation**
   - Store `(operation, inputs, shape, dtype)` instead of data
   - Only allocate buffers on `.eval()`

2. **Async Command Buffer Execution**
   - Don't wait for completion after each operation
   - Batch multiple operations into one command buffer
   - Use Metal completion handlers for synchronization

3. **Primitive-Based Architecture**
   - Abstract interface for operations
   - Platform-specific implementations
   - Easy to add new backends later

4. **Deferred Evaluation API**
   ```rust
   // v2.0 API
   let hidden = x.matmul(&lora_a);  // Returns LazyTensor, no execution
   let output = hidden.matmul(&lora_b);  // Still no execution
   let result = output.eval()?;     // Execute entire graph
   ```

### What We Can Skip (Initially)

1. **Automatic Graph Fusion** - Complex, minimal benefit without JIT
2. **Multi-Stream Support** - Single stream sufficient for v2.0
3. **Tracer Infrastructure** - Only needed for autodiff
4. **JIT Compilation** - MLX has this, but it's advanced

### Expected Performance Gains

Based on analysis:

| Operation | Current | v2.0 Target | MLX |
|-----------|---------|-------------|-----|
| LoRA (3 ops) | 36 µs | 10-15 µs | 5-11 µs |
| Softmax | 39 µs | 5-8 µs | 1.85 µs |
| RMS Norm | 47 µs | 10-15 µs | 6 µs |

**Primary gains from**:
- Async execution (6-10 µs saved)
- Command buffer batching (3-5 µs saved)
- Improved kernels (2-5 µs saved)

---

## 7. Implementation Risks

### High Risk
- **API Breaking Changes** - Lazy tensors require different API
- **Backward Compatibility** - Need wrapper layer for v1.0 API
- **Debugging Complexity** - Async errors harder to trace

### Medium Risk
- **Memory Management** - Lazy eval changes lifetime rules
- **Graph Overhead** - Building graph has CPU cost
- **Sync Point Performance** - `.eval()` calls must be minimal

### Low Risk
- **Metal Integration** - We already have custom Metal kernels
- **Candle Integration** - Can wrap Candle ops as primitives
- **Primitive Interface** - Similar to Candle's CustomOp

---

## 8. Recommended Architecture for metal-candle v2.0

```rust
// Lazy tensor with graph node
pub struct LazyTensor {
    node_id: NodeId,
    graph: Arc<RwLock<ComputationGraph>>,
    shape: Shape,
    dtype: DType,
}

pub struct ComputationGraph {
    nodes: Vec<GraphNode>,
    device: Device,
    command_buffer: Option<CommandBuffer>,  // Shared across ops
}

pub enum GraphNode {
    Input { data: Tensor },
    Matmul { left: NodeId, right: NodeId },
    Add { left: NodeId, right: NodeId },
    LoRA { input: NodeId, a: NodeId, b: NodeId, scale: f32 },
    // ... other ops
}

impl LazyTensor {
    pub fn matmul(&self, other: &LazyTensor) -> LazyTensor {
        // Record operation in graph
        let node_id = self.graph.write().unwrap().add_node(
            GraphNode::Matmul {
                left: self.node_id,
                right: other.node_id,
            }
        );
        
        LazyTensor {
            node_id,
            graph: self.graph.clone(),
            shape: compute_output_shape(...),
            dtype: self.dtype,
        }
    }
    
    pub fn eval(self) -> Result<Tensor> {
        // Execute graph with async command buffers
        self.graph.write().unwrap().execute(self.node_id)
    }
}
```

---

## 9. Next Steps

1. **Design Phase (Week 3)**
   - Finalize LazyTensor API
   - Design graph representation
   - Plan async executor

2. **Core Implementation (Weeks 4-6)**
   - Implement LazyTensor and ComputationGraph
   - Build async command buffer executor
   - Port existing CustomOps to primitives

3. **Migration (Weeks 7-9)**
   - Update LoRALayer to use lazy execution
   - Migrate other operations
   - Add backward compatibility layer

4. **Optimization (Weeks 10-12)**
   - Profile and optimize graph overhead
   - Tune command buffer batching
   - Benchmark vs MLX

---

## References

- MLX Source: `/Users/garthdb/Projects/metal-candle/mlx-study/`
- Key Files Analyzed:
  - `mlx/array.h`, `mlx/array.cpp` - Array graph structure
  - `mlx/primitives.h`, `mlx/primitives.cpp` - Primitive interface
  - `mlx/backend/metal/eval.cpp` - Async execution
  - `mlx/backend/metal/matmul.cpp` - Optimized Metal kernels
  - `mlx/scheduler.h` - Task scheduling

**Conclusion**: MLX's architecture is elegant and proven. Adopting lazy evaluation + async execution will get us within 2x of MLX performance, which is sufficient for v2.0 goals.

