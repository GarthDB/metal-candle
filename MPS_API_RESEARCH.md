# MPS API Research - Day 1

**Date**: December 10, 2024  
**Goal**: Understand Metal Performance Shaders APIs for matrix operations  
**Status**: 🔬 In Progress

---

## Overview

Metal Performance Shaders (MPS) provides highly optimized GPU kernels for common operations. We're researching which MPS operations can replace our custom kernels to achieve 5-20x speedups.

---

## Key MPS Operations for metal-candle

### 1. MPSMatrixMultiplication

**Purpose**: Matrix multiplication (critical for LoRA)

**Apple Documentation**: [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)

**Key Features**:
- Highly optimized for Apple Silicon
- Supports batched operations
- Configurable transpose modes
- Direct Metal buffer access

**API Surface**:
```objc
@interface MPSMatrixMultiplication : MPSKernel

// Initialize with dimensions
- (instancetype)initWithDevice:(id<MTLDevice>)device
                   transposeLeft:(BOOL)transposeLeft
                  transposeRight:(BOOL)transposeRight
                      resultRows:(NSUInteger)resultRows
                   resultColumns:(NSUInteger)resultColumns
                 interiorColumns:(NSUInteger)interiorColumns
                           alpha:(double)alpha
                            beta:(double)beta;

// Encode to command buffer
- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                   leftMatrix:(MPSMatrix *)leftMatrix
                  rightMatrix:(MPSMatrix *)rightMatrix
                 resultMatrix:(MPSMatrix *)resultMatrix;

@end
```

**Parameters**:
- `transposeLeft`: Apply transpose to left matrix
- `transposeRight`: Apply transpose to right matrix
- `resultRows`: M dimension
- `resultColumns`: N dimension
- `interiorColumns`: K dimension (must match for A×B)
- `alpha`: Scaling factor for product
- `beta`: Scaling factor for result accumulation

**Operation**: `result = alpha * (A × B) + beta * result`

**Rust FFI Mapping**:
```rust
pub struct MPSMatrixMultiplication {
    inner: *mut Object,
}

impl MPSMatrixMultiplication {
    pub fn new(
        device: &metal::Device,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_cols: usize,
        interior_cols: usize,
        alpha: f64,
        beta: f64,
    ) -> Result<Self>;
    
    pub fn encode(
        &self,
        command_buffer: &metal::CommandBufferRef,
        left: &MPSMatrix,
        right: &MPSMatrix,
        result: &MPSMatrix,
    );
}
```

**Expected Performance**:
- Current custom: 37-98 µs
- MLX (likely MPS): 5-11 µs
- **Expected with MPS**: 5-10 µs (5-10x speedup)

**Use Cases in metal-candle**:
- ✅ LoRA forward: `hidden = input @ lora_a`, `output = hidden @ lora_b`
- ✅ Attention: `scores = Q @ K.T`, `output = scores @ V`
- ✅ General matmul operations

---

### 2. MPSMatrixSoftMax

**Purpose**: Softmax normalization

**Apple Documentation**: [MPSMatrixSoftMax](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsoftmax)

**Key Features**:
- Numerically stable (handles max subtraction internally)
- Optimized for last dimension
- Single-pass algorithm

**API Surface**:
```objc
@interface MPSMatrixSoftMax : MPSMatrixUnaryKernel

- (instancetype)initWithDevice:(id<MTLDevice>)device;

- (void)encodeToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                  inputMatrix:(MPSMatrix *)inputMatrix
                 resultMatrix:(MPSMatrix *)resultMatrix;

@end
```

**Operation**: `result[i] = exp(input[i]) / sum(exp(input))`

**Numerical Stability**: MPS handles `max(input)` subtraction internally

**Rust FFI Mapping**:
```rust
pub struct MPSMatrixSoftMax {
    inner: *mut Object,
}

impl MPSMatrixSoftMax {
    pub fn new(device: &metal::Device) -> Result<Self>;
    
    pub fn encode(
        &self,
        command_buffer: &metal::CommandBufferRef,
        input: &MPSMatrix,
        result: &MPSMatrix,
    );
}
```

**Expected Performance**:
- Current custom: 39.4 µs
- MLX: 5.0 µs
- **Expected with MPS**: 3-5 µs (8-13x speedup)

**Use Cases**:
- ✅ Attention softmax
- ✅ Sampling probability distribution
- ✅ Any normalization requiring softmax

**Limitations**:
- May only support softmax along last dimension
- Need fallback for other dimensions

---

### 3. MPSMatrix (Descriptor & Container)

**Purpose**: Wrapper for Metal buffers representing matrices

**Apple Documentation**: [MPSMatrix](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)

**Key Features**:
- Wraps `MTLBuffer` with shape metadata
- Row-major layout (standard)
- Type information (float32, float16, etc.)

**API Surface**:
```objc
@interface MPSMatrixDescriptor : NSObject

+ (MPSMatrixDescriptor *)matrixDescriptorWithRows:(NSUInteger)rows
                                          columns:(NSUInteger)columns
                                         rowBytes:(NSUInteger)rowBytes
                                         dataType:(MPSDataType)dataType;

@property (readonly, nonatomic) NSUInteger rows;
@property (readonly, nonatomic) NSUInteger columns;
@property (readonly, nonatomic) NSUInteger rowBytes;
@property (readonly, nonatomic) MPSDataType dataType;

@end

@interface MPSMatrix : NSObject

- (instancetype)initWithBuffer:(id<MTLBuffer>)buffer
                    descriptor:(MPSMatrixDescriptor *)descriptor;

@property (readonly, nonatomic) id<MTLBuffer> data;
@property (readonly, nonatomic) NSUInteger rows;
@property (readonly, nonatomic) NSUInteger columns;

@end
```

**MPSDataType enum**:
```objc
typedef NS_ENUM(NSUInteger, MPSDataType) {
    MPSDataTypeFloat32 = 268435472,  // Our primary type
    MPSDataTypeFloat16 = 268435488,
    MPSDataTypeInt32   = 536870944,
};
```

**Rust FFI Mapping**:
```rust
pub struct MPSMatrixDescriptor {
    inner: *mut Object,
}

impl MPSMatrixDescriptor {
    pub fn new(
        rows: usize,
        cols: usize,
        row_bytes: usize,
        data_type: MPSDataType,
    ) -> Result<Self>;
}

pub struct MPSMatrix {
    inner: *mut Object,
}

impl MPSMatrix {
    pub fn new(
        buffer: &metal::Buffer,
        descriptor: &MPSMatrixDescriptor,
    ) -> Result<Self>;
    
    pub fn rows(&self) -> usize;
    pub fn columns(&self) -> usize;
    pub fn data(&self) -> &metal::Buffer;
}
```

**Integration with Candle**:
```rust
fn tensor_to_mps_matrix(tensor: &Tensor) -> Result<MPSMatrix> {
    // Extract Metal buffer from Candle tensor
    let storage = tensor.storage_and_layout();
    let metal_storage = match &*storage.0 {
        Storage::Metal(s) => s,
        _ => bail!("Tensor must be on Metal device"),
    };
    
    let buffer = metal_storage.buffer();
    let shape = tensor.shape();
    
    // Create descriptor
    let descriptor = MPSMatrixDescriptor::new(
        shape.dims()[0],
        shape.dims()[1],
        shape.dims()[1] * 4, // row_bytes for f32
        MPSDataType::Float32,
    )?;
    
    // Wrap in MPSMatrix
    MPSMatrix::new(buffer, &descriptor)
}
```

---

### 4. MPS for RMS Norm (Hybrid Approach)

**Challenge**: No direct `MPSRMSNorm` operation

**Options**:

#### Option A: Use MPSNNReduceFeatureChannelsMean + Custom
```objc
@interface MPSNNReduceFeatureChannelsMean : MPSNNReduceUnary
```

**Approach**:
1. MPS: Compute `mean(x²)` using reduce
2. Custom kernel: `output = x / sqrt(mean + eps)`

#### Option B: Custom Kernel (Optimized)
- Use insights from MPS patterns
- Optimize threadgroup usage
- Better than current, but not as fast as pure MPS

#### Option C: MPSCNNInstanceNormalization (Adapted)
```objc
@interface MPSCNNInstanceNormalization : MPSCNNNormalization
```

**Approach**: Instance normalization ≈ RMS norm with adaptations

**Recommendation**: Start with Option B (optimized custom), evaluate Option C if needed

**Expected Performance**:
- Current custom: 46.9 µs
- MLX: 5.0 µs
- **Expected with optimized**: 10-15 µs (3-5x speedup)
- **Expected with MPS hybrid**: 5-8 µs (6-9x speedup)

---

## MPS Memory Model

### Buffer Requirements

**Metal Buffer from Candle**:
- Candle tensors on Metal device already use `MTLBuffer`
- Can access directly via `MetalStorage::buffer()`
- No copy needed! Direct buffer sharing ✅

**Layout Requirements**:
- **Row-major**: MPS expects row-major layout (Candle default ✅)
- **Contiguous**: Must be contiguous in memory
- **Alignment**: 16-byte alignment for optimal performance

**Validation**:
```rust
fn validate_tensor_for_mps(tensor: &Tensor) -> Result<()> {
    // Check contiguous
    if !tensor.is_contiguous() {
        bail!("Tensor must be contiguous for MPS");
    }
    
    // Check device
    if !tensor.device().is_metal() {
        bail!("Tensor must be on Metal device");
    }
    
    // Check dtype
    if tensor.dtype() != DType::F32 {
        bail!("MPS currently only supports F32");
    }
    
    Ok(())
}
```

---

## Command Buffer Integration

### MPS Encoding Pattern

**Standard Flow**:
1. Get Metal command queue
2. Create command buffer
3. Encode MPS operations
4. Commit command buffer
5. Wait for completion

**Example**:
```rust
fn mps_matmul_example(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Get Metal device and command queue
    let device = a.device().as_metal_device()?;
    let queue = device.new_command_queue();
    let command_buffer = queue.new_command_buffer();
    
    // Convert tensors to MPS matrices
    let mps_a = tensor_to_mps_matrix(a)?;
    let mps_b = tensor_to_mps_matrix(b)?;
    
    // Create output buffer
    let output_shape = [a.dims()[0], b.dims()[1]];
    let output = create_output_tensor(device, &output_shape)?;
    let mps_output = tensor_to_mps_matrix(&output)?;
    
    // Create and encode MPS operation
    let matmul = MPSMatrixMultiplication::new(
        device,
        false, false, // no transpose
        output_shape[0], output_shape[1], a.dims()[1],
        1.0, 0.0, // alpha=1, beta=0
    )?;
    
    matmul.encode(command_buffer, &mps_a, &mps_b, &mps_output);
    
    // Execute
    command_buffer.commit();
    command_buffer.wait_until_completed();
    
    Ok(output)
}
```

**Performance Considerations**:
- Command buffer creation: ~1-2 µs
- Encoding: ~1 µs
- Execution: Depends on operation (5-10 µs for matmul)
- **Total overhead**: ~2-3 µs (acceptable)

---

## Rust FFI Strategy

### Required Dependencies

**Cargo.toml**:
```toml
[dependencies]
objc = { version = "0.2", optional = true }
metal = { version = "0.27", optional = true }

[features]
mps = ["dep:objc", "dep:metal"]
```

### FFI Safety Patterns

**Pattern 1: RAII Wrappers**
```rust
pub struct MPSMatrix {
    inner: *mut Object,
}

impl MPSMatrix {
    pub fn new(buffer: &metal::Buffer, descriptor: &MPSMatrixDescriptor) -> Result<Self> {
        unsafe {
            let class = class!(MPSMatrix);
            let obj: *mut Object = msg_send![class, alloc];
            let obj: *mut Object = msg_send![
                obj,
                initWithBuffer: buffer.as_ptr()
                descriptor: descriptor.as_ptr()
            ];
            
            if obj.is_null() {
                bail!("Failed to create MPSMatrix");
            }
            
            // Retain for ownership
            let _: () = msg_send![obj, retain];
            Ok(Self { inner: obj })
        }
    }
}

impl Drop for MPSMatrix {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}
```

**Pattern 2: Type-Safe Enums**
```rust
#[repr(u64)]
pub enum MPSDataType {
    Float32 = 268435472,
    Float16 = 268435488,
    Int32 = 536870944,
}
```

**Pattern 3: Error Handling**
```rust
fn check_mps_result(result: *mut Object) -> Result<*mut Object> {
    if result.is_null() {
        bail!("MPS operation failed");
    }
    Ok(result)
}
```

---

## Performance Projections

### Based on MPS Characteristics

| Operation | Current | MPS (Projected) | Speedup | Basis |
|-----------|---------|-----------------|---------|-------|
| **MatMul (512×512)** | 37 µs | 5-7 µs | 5-7x | MLX data |
| **MatMul (1024×1024)** | 55 µs | 6-8 µs | 7-9x | Scales well |
| **MatMul (2048×2048)** | 98 µs | 8-12 µs | 8-12x | Better for large |
| **Softmax** | 39 µs | 3-5 µs | 8-13x | Highly optimized |
| **RMS Norm** | 47 µs | 5-8 µs | 6-9x | Hybrid approach |

**Confidence Levels**:
- MatMul: **High** (MLX uses MPS or equivalent, we have data)
- Softmax: **High** (Simple operation, MPS is perfect fit)
- RMS Norm: **Medium** (May need hybrid approach)

---

## Risks & Mitigation

### Risk 1: MPS API Changes

**Likelihood**: Low  
**Impact**: Medium  
**Mitigation**: 
- Use stable APIs (MPSMatrixMultiplication has been stable for years)
- Document minimum macOS version (14.0+)
- Test on multiple OS versions

### Risk 2: Dimension Limitations

**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**:
- Keep custom kernels as fallback
- Document supported shapes
- Auto-select based on shape

### Risk 3: Memory Layout Issues

**Likelihood**: Medium  
**Impact**: High  
**Mitigation**:
- Validate contiguity before MPS
- Auto-convert if needed (`.contiguous()`)
- Clear error messages

### Risk 4: Performance Not As Expected

**Likelihood**: Low  
**Impact**: High  
**Mitigation**:
- Early prototyping (Day 5)
- Benchmark before full migration
- Have rollback plan

---

## Next Steps (Day 2)

### Immediate Actions

1. **Create FFI module structure**:
   ```bash
   mkdir -p src/backend/mps
   touch src/backend/mps/mod.rs
   touch src/backend/mps/ffi.rs
   touch src/backend/mps/matrix.rs
   ```

2. **Implement MPSMatrixDescriptor wrapper**:
   - Safe Rust wrapper
   - Memory management (retain/release)
   - Type conversions

3. **Implement MPSMatrix wrapper**:
   - Integration with Metal buffers
   - Conversion from Candle tensors
   - Shape validation

4. **Test basic FFI**:
   - Create simple test
   - Verify no memory leaks
   - Confirm correct initialization

### Success Criteria for Day 2

- ✅ FFI module compiles
- ✅ Can create MPSMatrixDescriptor from Rust
- ✅ Can create MPSMatrix from Metal buffer
- ✅ No memory leaks in Instruments
- ✅ Basic correctness test passes

---

## References

### Apple Documentation

- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [MPSMatrixMultiplication](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixmultiplication)
- [MPSMatrixSoftMax](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixsoftmax)
- [MPSMatrix](https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrix)
- [Metal Programming Guide](https://developer.apple.com/metal/)

### Rust FFI

- [objc crate](https://docs.rs/objc/latest/objc/)
- [metal-rs](https://docs.rs/metal/latest/metal/)
- [The Rust FFI Omnibus](http://jakegoulding.com/rust-ffi-omnibus/)

### Related Projects

- [MLX (likely uses MPS)](https://github.com/ml-explore/mlx)
- [PyTorch Metal Backend](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/mps)
- [Metal-rs Examples](https://github.com/gfx-rs/metal-rs/tree/master/examples)

---

**Day 1 Status**: ✅ Research Complete  
**Key Finding**: MPS provides the operations we need with expected 5-20x speedups  
**Next**: Day 2 - Rust FFI bindings implementation  
**Confidence**: High - MPS is the right path to MLX parity

