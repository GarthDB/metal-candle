//! Integration tests for lazy execution and computation graphs.

#![cfg(feature = "graph")]

use candle_core::{DType, Device, Tensor};
use metal_candle::graph::LazyTensor;

#[test]
fn test_lazy_tensor_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create lazy tensors (no computation)
    let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0], &[3], &device)?;
    let b = LazyTensor::from_slice(&[4.0, 5.0, 6.0], &[3], &device)?;

    // Operations should not execute immediately
    assert_eq!(a.graph_size(), 1);
    assert_eq!(b.graph_size(), 1);

    // Build computation graph (still no execution)
    let c = a.add(&b)?;
    assert_eq!(c.graph_size(), 3); // a, b, and add node

    // Evaluate
    let result = c.eval()?;
    let expected = vec![5.0, 7.0, 9.0];
    assert_eq!(result.to_vec1::<f32>()?, expected);

    Ok(())
}

#[test]
fn test_lazy_tensor_chain() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = LazyTensor::from_slice(&[1.0, 2.0], &[2], &device)?;
    let b = LazyTensor::from_slice(&[3.0, 4.0], &[2], &device)?;

    // Chain operations
    let c = a.add(&b)?;
    let d = c.mul_scalar(2.0)?;

    // Should have 4 nodes: a, b, add, mul_scalar
    assert_eq!(d.graph_size(), 4);

    let result = d.eval()?;
    let expected = vec![8.0, 12.0]; // (1+3)*2, (2+4)*2
    assert_eq!(result.to_vec1::<f32>()?, expected);

    Ok(())
}

#[test]
fn test_lazy_matmul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create 2x3 and 3x2 matrices
    let a = LazyTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    let b = LazyTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)?;

    let c = a.matmul(&b)?;

    // Output should be 2x2
    assert_eq!(c.shape().dims(), &[2, 2]);

    let result = c.eval()?;

    // Verify matmul result
    let result_vec = result.to_vec2::<f32>()?;
    assert_eq!(result_vec[0], vec![22.0, 28.0]);
    assert_eq!(result_vec[1], vec![49.0, 64.0]);

    Ok(())
}

#[test]
fn test_lazy_vs_eager_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Same computation lazy and eager
    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![5.0, 6.0, 7.0, 8.0];

    // Lazy
    let a_lazy = LazyTensor::from_slice(&a_data, &[4], &device)?;
    let b_lazy = LazyTensor::from_slice(&b_data, &[4], &device)?;
    let c_lazy = a_lazy.add(&b_lazy)?.mul_scalar(2.0)?;
    let result_lazy = c_lazy.eval()?;

    // Eager (using Candle directly)
    let a_eager = Tensor::from_slice(&a_data, &[4], &device)?;
    let b_eager = Tensor::from_slice(&b_data, &[4], &device)?;
    let c_eager = a_eager.add(&b_eager)?.affine(2.0, 0.0)?;

    // Compare results
    let lazy_vec = result_lazy.to_vec1::<f32>()?;
    let eager_vec = c_eager.to_vec1::<f32>()?;
    assert_eq!(lazy_vec, eager_vec);

    Ok(())
}

#[test]
fn test_zeros_and_ones() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let zeros = LazyTensor::zeros(&[2, 3], DType::F32, &device)?;
    let ones = LazyTensor::ones(&[2, 3], DType::F32, &device)?;

    let result = zeros.add(&ones)?.eval()?;

    // Should be all ones
    let result_vec = result.to_vec2::<f32>()?;
    assert_eq!(result_vec, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);

    Ok(())
}
