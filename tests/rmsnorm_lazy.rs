//! Tests for RMS Norm lazy execution.

#![cfg(feature = "graph")]

use candle_core::{Device, Tensor};
use metal_candle::graph::LazyTensor;

#[test]
fn test_rmsnorm_lazy_basic() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create input tensor
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input = Tensor::from_slice(&input_data, &[4], &device)?;

    // Eager execution
    let alpha = Tensor::ones(&[4], input.dtype(), &device)?;
    let eager_output = candle_nn::ops::rms_norm(&input, &alpha, 1e-5)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let output_lazy = input_lazy.rms_norm(1e-5)?;
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
fn test_rmsnorm_lazy_2d() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create 2D input tensor [batch, features]
    let input = Tensor::randn(0f32, 1f32, &[4, 64], &device)?;

    // Eager execution
    let last_dim = input.dims()[input.dims().len() - 1];
    let alpha = Tensor::ones(&[last_dim], input.dtype(), &device)?;
    let eager_output = candle_nn::ops::rms_norm(&input, &alpha, 1e-5)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let lazy_output = input_lazy.rms_norm(1e-5)?.eval()?;

    // Validate
    assert_eq!(eager_output.dims(), &[4, 64]);
    assert_eq!(lazy_output.dims(), &[4, 64]);

    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_rmsnorm_lazy_batched() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Batched 3D tensor [batch, seq_len, features]
    let input = Tensor::randn(0f32, 1f32, &[2, 10, 128], &device)?;

    // Eager execution
    let last_dim = input.dims()[input.dims().len() - 1];
    let alpha = Tensor::ones(&[last_dim], input.dtype(), &device)?;
    let eager_output = candle_nn::ops::rms_norm(&input, &alpha, 1e-5)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input)?;
    let lazy_output = input_lazy.rms_norm(1e-5)?.eval()?;

    // Validate
    assert_eq!(eager_output.dims(), &[2, 10, 128]);
    assert_eq!(lazy_output.dims(), &[2, 10, 128]);

    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_rmsnorm_lazy_different_eps() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test with different epsilon values
    let input = Tensor::randn(0f32, 1f32, &[8, 32], &device)?;

    for eps in [1e-8, 1e-6, 1e-5, 1e-4] {
        let last_dim = input.dims()[input.dims().len() - 1];
        let alpha = Tensor::ones(&[last_dim], input.dtype(), &device)?;
        let eager_output = candle_nn::ops::rms_norm(&input, &alpha, eps)?;

        let input_lazy = LazyTensor::from_tensor(input.clone())?;
        let lazy_output = input_lazy.rms_norm(eps)?.eval()?;

        let diff = (eager_output - lazy_output)?.abs()?;
        let diff_flat = diff.flatten_all()?;
        assert!(
            diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4,
            "Failed for eps {}",
            eps
        );
    }

    Ok(())
}

#[test]
fn test_rmsnorm_lazy_normalization_property() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // RMS norm should approximately normalize the RMS to 1
    let input = Tensor::randn(0f32, 10f32, &[100], &device)?;

    let input_lazy = LazyTensor::from_tensor(input)?;
    let output_lazy = input_lazy.rms_norm(1e-5)?.eval()?;

    // Compute RMS of output
    let squared = output_lazy.sqr()?;
    let mean = squared.mean(0)?;
    let rms = mean.sqrt()?.to_scalar::<f32>()?;

    println!("Output RMS: {:.6}", rms);

    // RMS should be close to 1.0 (within reasonable tolerance)
    assert!(
        (rms - 1.0).abs() < 0.1,
        "RMS norm output RMS {} should be close to 1.0",
        rms
    );

    Ok(())
}

#[test]
fn test_rmsnorm_lazy_chain() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Test chaining RMS norm with other operations
    let input = Tensor::randn(0f32, 1f32, &[4, 64], &device)?;

    // Lazy: input -> rms_norm -> mul_scalar
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let normed = input_lazy.rms_norm(1e-5)?;
    let scaled = normed.mul_scalar(2.0)?;
    let lazy_output = scaled.eval()?;

    // Eager: same operations
    let last_dim = input.dims()[input.dims().len() - 1];
    let alpha = Tensor::ones(&[last_dim], input.dtype(), &device)?;
    let normed_eager = candle_nn::ops::rms_norm(&input, &alpha, 1e-5)?;
    let eager_output = normed_eager.affine(2.0, 0.0)?;

    // Compare
    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_rmsnorm_lazy_with_matmul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Common pattern: matmul -> rms_norm
    let input = Tensor::randn(0f32, 1f32, &[4, 32], &device)?;
    let weight = Tensor::randn(0f32, 1f32, &[32, 64], &device)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let weight_lazy = input_lazy.add_tensor_to_graph(weight.clone())?;
    let hidden = input_lazy.matmul(&weight_lazy)?;
    let normed = hidden.rms_norm(1e-5)?;
    let lazy_output = normed.eval()?;

    // Eager execution
    let hidden_eager = input.matmul(&weight)?;
    let last_dim = hidden_eager.dims()[hidden_eager.dims().len() - 1];
    let alpha = Tensor::ones(&[last_dim], hidden_eager.dtype(), &device)?;
    let eager_output = candle_nn::ops::rms_norm(&hidden_eager, &alpha, 1e-5)?;

    // Compare
    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}
