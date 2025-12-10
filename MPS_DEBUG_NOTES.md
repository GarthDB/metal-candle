# MPS Debugging Notes

## Problem

MPS matmul produces incorrect results:
- Input A: `[[1, 2, 3], [4, 5, 6]]`
- Input B: `[[7, 8], [9, 10], [11, 12]]`
- Expected: `[[58, 64], [139, 154]]`
- Actual: `[[1528, 0], [5696, 0]]`

## Key Issues

1. **Second column is all zeros** - suggests stride/offset problem
2. **First column values are wrong** - way too large (1528 vs 58)
3. **Command buffer issue is FIXED** - No more Metal assertion!

## Current Approach (WRONG)

We're creating MPS matrices directly from Candle's Metal buffers:

```rust
let left_buffer = left_storage.buffer();  // Arc<metal::Buffer>
let mps_left = MPSMatrix::new(left_buffer, &left_desc)?;
```

**Problem**: We're ignoring:
- `left_layout.start_offset()` - Buffer may not start at offset 0
- `left_layout.stride()` - Actual memory layout (row-major vs column-major)
- Contiguity - Tensor may not be contiguous in memory

## What We Need

MPS's `MPSMatrixDescriptor` needs:
1. **Correct row_bytes**: Actual bytes between rows in memory
2. **Buffer offset**: Where the data actually starts
3. **Data layout**: Must be contiguous and row-major

## Solution

**Option 1**: Ensure contiguity (RECOMMENDED)
```rust
// Make tensor contiguous before passing to MPS
let left_contiguous = left_tensor.contiguous()?;
let right_contiguous = right_tensor.contiguous()?;

// Now we can safely use the buffer
let left_storage = ... // from contiguous tensor
let left_buffer = left_storage.buffer();
```

**Option 2**: Use buffer offset + custom stride
```rust
// Account for layout offset
let offset = left_layout.start_offset();
let stride = left_layout.stride();

// Create MPSMatrixDescriptor with offset
// But MPS doesn't support buffer offsets directly!
// Would need to offset the buffer pointer (unsafe)
```

**Recommendation**: Option 1 - ensure contiguity

## Next Step

Modify `custom_matmul.rs` to call `.contiguous()` on input tensors before extracting buffers.

