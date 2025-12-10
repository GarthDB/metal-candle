//! Tests for Softmax lazy execution.

#![cfg(feature = "graph")]

use candle_core::{Device, Tensor};
use metal_candle::graph::LazyTensor;

#[test]
fn test_softmax_lazy_basic() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create input tensor
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_slice(&input_data, &[4], &device)?;

    // Eager execution
    let eager_output = candle_nn::ops::softmax(&input, 0)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let output_lazy = input_lazy.softmax(0)?;
    let lazy_output = output_lazy.eval()?;

    // Validate shapes match
    assert_eq!(eager_output.shape(), lazy_output.shape());

    // Validate results are close
    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    let max_diff = diff_flat.max(0)?.to_scalar::<f32>()?;
    println!("Max difference: {:.6e}", max_diff);
    assert!(
        max_diff < 1e-4,
        "Max difference {} exceeds threshold",
        max_diff
    );

    Ok(())
}

#[test]
fn test_softmax_lazy_2d() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create 2D input tensor
    let input = Tensor::randn(0f32, 1f32, &[4, 8], &device)?;

    // Eager execution along last dimension
    let eager_output = candle_nn::ops::softmax(&input, 1)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let lazy_output = input_lazy.softmax(1)?.eval()?;

    // Validate
    assert_eq!(eager_output.dims(), &[4, 8]);
    assert_eq!(lazy_output.dims(), &[4, 8]);

    let diff = (eager_output - lazy_output.clone())?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    // Verify softmax property: sum along dim should be ~1
    let sum = lazy_output.sum(1)?;
    let sum_vec = sum.to_vec1::<f32>()?;
    for s in sum_vec {
        assert!((s - 1.0).abs() < 1e-5, "Softmax sum {} should be 1.0", s);
    }

    Ok(())
}

#[test]
fn test_softmax_lazy_batched() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Batched 3D tensor [batch, seq_len, features]
    let input = Tensor::randn(0f32, 1f32, &[2, 10, 64], &device)?;

    // Eager execution along last dimension
    let eager_output = candle_nn::ops::softmax(&input, 2)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let lazy_output = input_lazy.softmax(2)?.eval()?;

    // Validate
    assert_eq!(eager_output.dims(), &[2, 10, 64]);
    assert_eq!(lazy_output.dims(), &[2, 10, 64]);

    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_softmax_lazy_different_dims() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test softmax along different dimensions
    let input = Tensor::randn(0f32, 1f32, &[3, 4, 5], &device)?;

    for dim in 0..3 {
        let eager_output = candle_nn::ops::softmax(&input, dim)?;

        let input_lazy = LazyTensor::from_tensor(input.clone())?;
        let lazy_output = input_lazy.softmax(dim)?.eval()?;

        let diff = (eager_output - lazy_output.clone())?.abs()?;
        let diff_flat = diff.flatten_all()?;
        assert!(
            diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4,
            "Failed for dim {}",
            dim
        );

        // Verify softmax sum property
        let sum = lazy_output.sum(dim)?;
        let expected_shape = match dim {
            0 => vec![4, 5],
            1 => vec![3, 5],
            2 => vec![3, 4],
            _ => unreachable!(),
        };
        assert_eq!(sum.dims(), expected_shape.as_slice());
    }

    Ok(())
}

#[test]
fn test_softmax_lazy_numerical_stability() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test with large values (should be numerically stable)
    let input_data: Vec<f32> = vec![100.0, 200.0, 300.0, 400.0];
    let input = Tensor::from_slice(&input_data, &[4], &device)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let output_lazy = input_lazy.softmax(0)?.eval()?;

    // Verify no NaN or Inf
    let output_vec = output_lazy.to_vec1::<f32>()?;
    for v in &output_vec {
        assert!(v.is_finite(), "Softmax produced non-finite value: {}", v);
    }

    // Verify sum is 1
    let sum: f32 = output_vec.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax sum {} should be 1.0",
        sum
    );

    // Largest value should have probability close to 1
    assert!(
        output_vec[3] > 0.99,
        "Largest input should have probability ~1"
    );

    Ok(())
}

#[test]
fn test_softmax_lazy_chain() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test chaining softmax with other operations
    let input = Tensor::randn(0f32, 1f32, &[4, 8], &device)?;

    // Lazy: input -> softmax -> mul_scalar
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let soft = input_lazy.softmax(1)?;
    let scaled = soft.mul_scalar(2.0)?;
    let lazy_output = scaled.eval()?;

    // Eager: same operations
    let soft_eager = candle_nn::ops::softmax(&input, 1)?;
    let eager_output = soft_eager.affine(2.0, 0.0)?;

    // Compare
    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}
