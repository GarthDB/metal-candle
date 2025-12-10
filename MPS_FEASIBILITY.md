# MPS Feasibility Assessment

## Key Finding

**metal-rs MPS module**: ❌ Only has ray tracing APIs, NOT matrix/NN operations  
**Needed**: `MPSMatrixMultiplication`, `MPSMatrixSoftMax`, etc.  
**Solution**: Use `objc` crate directly to call MPS APIs

## What metal-rs Provides

Looking at `/metal-0.27.0/src/mps.rs`:
- ✅ `MPSRayIntersector` - Ray tracing
- ✅ `MPSAccelerationStructure` - Ray tracing geometry
- ❌ No matrix operations
- ❌ No neural network operations  
- ❌ No MPSMatrixMultiplication
- ❌ No MPSMatrixSoftMax

**Conclusion**: metal-rs MPS bindings are incomplete for ML use cases

## What We Need

From Apple's Metal Performance Shaders framework:

### Matrix Operations
- `MPSMatrix` - Wrapper around Metal buffers
- `MPSMatrixDescriptor` - Describes matrix layout
- `MPSMatrixMultiplication` - C = αAB + βC
- `MPSMatrixSoftMax` - Softmax operation

### Neural Network Operations (Optional)
- `MPSMatrixNeuron` - Activation functions
- `MPSNNGraph` - Graph-based execution (too high-level)

## Approach: Direct objc Bindings

Since metal-rs doesn't provide these, we create our own bindings:

### Step 1: Define Objective-C Classes

```rust
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};

// MPSMatrixDescriptor
#[link(name = "MetalPerformanceShaders", kind = "framework")]
extern "C" {}

fn get_mps_matrix_descriptor_class() -> &'static Class {
    Class::get("MPSMatrixDescriptor").expect("MPSMatrixDescriptor class not found")
}

fn get_mps_matrix_multiplication_class() -> &'static Class {
    Class::get("MPSMatrixMultiplication").expect("MPSMatrixMultiplication class not found")
}
```

### Step 2: Create MPSMatrix Wrappers

```rust
pub struct MPSMatrixDescriptor {
    ptr: *mut Object,
}

impl MPSMatrixDescriptor {
    pub fn new(
        rows: usize,
        columns: usize,
        row_bytes: usize,
        data_type: MPSDataType,
    ) -> Self {
        unsafe {
            let class = get_mps_matrix_descriptor_class();
            let desc: *mut Object = msg_send![class, 
                matrixDescriptorWithRows: rows as u64
                columns: columns as u64
                rowBytes: row_bytes as u64
                dataType: data_type as u32
            ];
            Self { ptr: desc }
        }
    }
}

pub struct MPSMatrix {
    ptr: *mut Object,
}

impl MPSMatrix {
    pub fn from_buffer(
        buffer: &metal::BufferRef,
        descriptor: &MPSMatrixDescriptor,
    ) -> Self {
        unsafe {
            let class = Class::get("MPSMatrix").expect("MPSMatrix not found");
            let matrix: *mut Object = msg_send![class, alloc];
            let matrix: *mut Object = msg_send![matrix,
                initWithBuffer: buffer
                descriptor: descriptor.ptr
            ];
            Self { ptr: matrix }
        }
    }
}
```

### Step 3: MPSMatrixMultiplication

```rust
pub struct MPSMatrixMultiplication {
    ptr: *mut Object,
}

impl MPSMatrixMultiplication {
    pub fn new(
        device: &metal::DeviceRef,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Self {
        unsafe {
            let class = get_mps_matrix_multiplication_class();
            let mul: *mut Object = msg_send![class, alloc];
            let mul: *mut Object = msg_send![mul,
                initWithDevice: device
                resultRows: result_rows as u64
                resultColumns: result_columns as u64
                interiorColumns: interior_columns as u64
                alpha: alpha
                beta: beta
            ];
            Self { ptr: mul }
        }
    }

    pub fn encode(
        &self,
        command_buffer: &metal::CommandBufferRef,
        left_matrix: &MPSMatrix,
        right_matrix: &MPSMatrix,
        result_matrix: &MPSMatrix,
    ) {
        unsafe {
            let _: () = msg_send![self.ptr,
                encodeToCommandBuffer: command_buffer
                leftMatrix: left_matrix.ptr
                rightMatrix: right_matrix.ptr
                resultMatrix: result_matrix.ptr
            ];
        }
    }
}
```

## Complexity Assessment

### Low Complexity ✅
- Creating bindings for specific MPS classes
- Basic wrapper structs
- Simple objc msg_send calls

### Medium Complexity ⚠️
- Memory management (retain/release)
- Error handling
- Data type mapping

### High Complexity ❌
- Full MPS API coverage
- Safe Rust wrappers
- Upstream contribution to metal-rs

## Estimated Effort

**Minimal Prototype** (for benchmarking):
- Time: 2-4 hours
- Scope: MPSMatrixMultiplication only
- Goal: Prove 5-20x speedup

**Production Integration**:
- Time: 1-2 weeks
- Scope: Matrix ops + Softmax
- Goal: Replace custom kernels where beneficial

**Complete MPS Bindings** (upstream to metal-rs):
- Time: 4-6 weeks
- Scope: Full MPS matrix/NN APIs
- Goal: Community contribution

## Decision Point

### Option A: Quick Prototype (RECOMMENDED)

**Goal**: Prove MPS performance in 2-4 hours

**Deliverable**:
- Simple matmul benchmark
- MPS vs custom kernel vs MLX
- Go/No-Go decision

**If 5x+ speedup**: Proceed to production
**If < 3x speedup**: Document and move on

### Option B: Production Implementation

**Prerequisites**: Option A shows promise

**Scope**:
- MPSMatrixMultiplication for LoRA
- MPSMatrixSoftMax
- Error handling + tests

**Time**: 1-2 weeks

### Option C: Full MPS Bindings

**Prerequisites**: Option B successful

**Scope**:
- Complete MPS matrix API
- Contribute to metal-rs
- Documentation

**Time**: 4-6 weeks

## Next Steps

1. ✅ **Create minimal MPS matmul bindings** (2 hours)
2. ✅ **Benchmark vs custom kernel** (30 min)
3. ✅ **Compare vs MLX** (30 min)
4. ✅ **Document results** (30 min)

**Total**: 3-4 hours to decision point

---

**Status**: Ready for prototype  
**Blocker**: None (objc crate available)  
**Risk**: Low (worst case: doesn't help, we learn)

