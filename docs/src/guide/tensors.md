# Tensor Operations

Working with tensors in metal-candle using the Candle framework.

## Creating Tensors

### From Rust Data

```rust
use candle_core::{Tensor, DType, Device};

// From scalar
let scalar = Tensor::new(42.0f32, &device)?;

// From vector
let vec_data = vec![1.0, 2.0, 3.0, 4.0];
let tensor = Tensor::from_vec(vec_data, (2, 2), &device)?;

// From 2D vector  
let data_2d = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
let tensor = Tensor::from_vec2(&data_2d, &device)?;
```

### Common Initializations

```rust
// Zeros
let zeros = Tensor::zeros((3, 4), DType::F32, &device)?;

// Ones
let ones = Tensor::ones((3, 4), DType::F32, &device)?;

// Random (uniform)
let random = Tensor::rand(0.0f32, 1.0f32, (3, 4), &device)?;

// Random (normal)
let normal = Tensor::randn(0.0f32, 1.0f32, (3, 4), &device)?;
```

## Tensor Properties

```rust
let tensor = Tensor::zeros((2, 3, 4), DType::F32, &device)?;

// Shape
println!("Shape: {:?}", tensor.shape());  // [2, 3, 4]

// Dimensions
println!("Dims: {}", tensor.dims().len());  // 3

// Data type
println!("DType: {:?}", tensor.dtype());  // F32

// Device
println!("Device: {:?}", tensor.device());  // Metal/CPU

// Number of elements
println!("Elements: {}", tensor.elem_count());  // 24
```

## Basic Operations

### Element-wise Operations

```rust
let a = Tensor::ones((2, 2), DType::F32, &device)?;
let b = Tensor::ones((2, 2), DType::F32, &device)?;

// Addition
let c = a.add(&b)?;  // or: (&a + &b)?

// Subtraction
let c = a.sub(&b)?;

// Multiplication
let c = a.mul(&b)?;

// Division
let c = a.div(&b)?;
```

### Matrix Operations

```rust
// Matrix multiplication
let a = Tensor::randn(0.0f32, 1.0f32, (2, 3), &device)?;
let b = Tensor::randn(0.0f32, 1.0f32, (3, 4), &device)?;
let c = a.matmul(&b)?;  // Shape: [2, 4]

// Transpose
let at = a.transpose(0, 1)?;  // Swap dimensions 0 and 1

// Make contiguous after transpose (important for Metal!)
let at_contig = at.contiguous()?;
```

## Reshaping & Indexing

### Reshape

```rust
let tensor = Tensor::arange(0f32, 12f32, &device)?;
println!("Original: {:?}", tensor.shape());  // [12]

// Reshape to 2D
let reshaped = tensor.reshape((3, 4))?;
println!("Reshaped: {:?}", reshaped.shape());  // [3, 4]

// Reshape to 3D
let reshaped_3d = tensor.reshape((2, 2, 3))?;
println!("3D: {:?}", reshaped_3d.shape());  // [2, 2, 3]
```

### Indexing

```rust
let tensor = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;

// Index single element
let elem = tensor.i((0, 1))?;  // Row 0, Column 1

// Index range
let row = tensor.i(0)?;  // First row

// Slice
let slice = tensor.narrow(0, 0, 2)?;  // First 2 rows
```

### Squeeze & Unsqueeze

```rust
// Add dimension
let expanded = tensor.unsqueeze(0)?;  // Add batch dimension

// Remove dimension
let squeezed = expanded.squeeze(0)?;  // Remove batch dimension
```

## Aggregation Operations

### Reduction

```rust
let tensor = Tensor::arange(0f32, 12f32, &device)?.reshape((3, 4))?;

// Sum
let total = tensor.sum_all()?;  // Sum all elements

// Sum along dimension
let col_sums = tensor.sum(0)?;  // Sum columns

// Mean
let mean = tensor.mean_all()?;

// Max/Min
let max_val = tensor.max(0)?;
let min_val = tensor.min(0)?;
```

## Advanced Operations

### Broadcasting

```rust
// Tensors with compatible shapes broadcast automatically
let a = Tensor::ones((3, 1), DType::F32, &device)?;
let b = Tensor::ones((1, 4), DType::F32, &device)?;

let c = a.add(&b)?;  // Result shape: [3, 4]
```

### Concatenation

```rust
// Concatenate along dimension
let a = Tensor::ones((2, 3), DType::F32, &device)?;
let b = Tensor::zeros((2, 3), DType::F32, &device)?;

let stacked = Tensor::cat(&[&a, &b], 0)?;  // Shape: [4, 3]
```

### Chunking

```rust
// Split tensor into chunks
let tensor = Tensor::arange(0f32, 12f32, &device)?;
let chunks = tensor.chunk(3, 0)?;  // Split into 3 chunks

for (i, chunk) in chunks.iter().enumerate() {
    println!("Chunk {}: {:?}", i, chunk.shape());
}
```

## Neural Network Operations

### Activation Functions

```rust
use candle_nn::ops;

// ReLU
let activated = ops::relu(&tensor)?;

// Softmax
let probs = ops::softmax(&logits, -1)?;  // Last dimension

// Layer Normalization
let normalized = ops::layer_norm(&tensor, &weight, &bias, 1e-5)?;
```

### Dropout

```rust
use candle_nn::Dropout;

let dropout = Dropout::new(0.1);  // 10% dropout rate
let output = dropout.forward(&tensor, true)?;  // true = training mode
```

## Type Conversions

### Data Type

```rust
// Convert to different precision
let f32_tensor = tensor.to_dtype(DType::F32)?;
let f16_tensor = tensor.to_dtype(DType::F16)?;
```

### Device Transfer

```rust
// Move between devices
let gpu = Device::new_metal(0)?;
let cpu = Device::Cpu;

let tensor_gpu = tensor.to_device(&gpu)?;
let tensor_cpu = tensor_gpu.to_device(&cpu)?;
```

### To/From Rust Data

```rust
// Tensor to Vec
let vec_data: Vec<f32> = tensor.to_vec1()?;  // 1D
let vec2_data: Vec<Vec<f32>> = tensor.to_vec2()?;  // 2D

// Get scalar value
let scalar_value: f32 = scalar_tensor.to_scalar()?;
```

## Important: Metal Specifics

### Contiguous Tensors

Metal operations require contiguous tensors:

```rust
// After transpose, make contiguous
let transposed = tensor.transpose(0, 1)?;
let ready_for_metal = transposed.contiguous()?;

// Now safe to use in Metal operations
let result = ready_for_metal.matmul(&other)?;
```

### Supported Data Types

Metal supports:
- ✅ `DType::F16` (recommended)
- ✅ `DType::F32`
- ❌ `DType::F64` (not supported - use CPU)

```rust
// Good for Metal
let tensor = Tensor::zeros((10, 10), DType::F16, &metal_device)?;

// Error on Metal!
// let tensor = Tensor::zeros((10, 10), DType::F64, &metal_device)?;
```

## Performance Tips

### 1. Keep Operations on Same Device

```rust
// Good: Everything on GPU
let a = Tensor::ones((1024, 1024), DType::F32, &gpu)?;
let b = Tensor::ones((1024, 1024), DType::F32, &gpu)?;
let c = a.matmul(&b)?;  // Fast!

// Bad: Mixing devices
let a = Tensor::ones((1024, 1024), DType::F32, &gpu)?;
let b = Tensor::ones((1024, 1024), DType::F32, &cpu)?;
let c = a.matmul(&b)?;  // Slow! Transfers b to GPU
```

### 2. Use Appropriate Precision

```rust
// For training/inference on Metal
let tensor = Tensor::zeros(shape, DType::F16, &device)?;  // Faster, less memory

// For high precision needed
let tensor = Tensor::zeros(shape, DType::F32, &device)?;
```

### 3. Reuse Tensors

```rust
// Preallocate and reuse
let mut buffer = Tensor::zeros(shape, dtype, &device)?;

for item in data {
    // Reuse buffer instead of allocating new
    buffer = process(&buffer, item)?;
}
```

## Common Patterns

### Batch Processing

```rust
// Stack individual samples into batch
let samples = vec![sample1, sample2, sample3, sample4];
let batch = Tensor::stack(&samples, 0)?;  // Batch dimension = 0
```

### Normalization

```rust
// L2 normalization
let l2_norm = tensor.sqr()?.sum_keepdim(-1)?.sqrt()?;
let normalized = tensor.broadcast_div(&l2_norm)?;
```

### Masking

```rust
// Create attention mask
let mask = Tensor::ones((seq_len, seq_len), DType::F32, &device)?;
let causal_mask = mask.triu(1)?;  // Upper triangular

// Apply mask
let masked = scores.masked_fill(&causal_mask, f32::NEG_INFINITY)?;
```

## Error Handling

```rust
use candle_core::Error as CandleError;

match tensor.matmul(&other) {
    Ok(result) => println!("Success: {:?}", result.shape()),
    Err(CandleError::ShapeMismatch { .. }) => {
        eprintln!("Shape mismatch in matrix multiplication");
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## See Also

- [Candle Documentation](https://github.com/huggingface/candle)
- [Device Management](./devices.md)
- [Performance Tips](../architecture/performance.md)
