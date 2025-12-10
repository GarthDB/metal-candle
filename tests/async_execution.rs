//! Tests for async execution (Phase 5).

#![cfg(feature = "async-exec")]

use candle_core::{Device, Tensor};
use metal_candle::graph::LazyTensor;

#[tokio::test]
async fn test_async_eval_basic() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create lazy tensors with explicit F32
    let a_tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let b_tensor = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device)?;

    let a = LazyTensor::from_tensor(a_tensor)?;
    let b = a.add_tensor_to_graph(b_tensor)?;

    // Build graph
    let c = a.add(&b)?;

    // Evaluate asynchronously
    let result = c.eval_async().await?;
    let expected = vec![5.0, 7.0, 9.0];
    assert_eq!(result.to_vec1::<f32>()?, expected);

    Ok(())
}

#[tokio::test]
async fn test_async_vs_sync_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![5.0f32, 6.0, 7.0, 8.0];

    // Create two identical graphs
    let a1 = LazyTensor::from_tensor(Tensor::from_slice(&a_data, &[4], &device)?)?;
    let b1 = a1.add_tensor_to_graph(Tensor::from_slice(&b_data, &[4], &device)?)?;
    let c1 = a1.add(&b1)?.mul_scalar(2.0)?;

    let a2 = LazyTensor::from_tensor(Tensor::from_slice(&a_data, &[4], &device)?)?;
    let b2 = a2.add_tensor_to_graph(Tensor::from_slice(&b_data, &[4], &device)?)?;
    let c2 = a2.add(&b2)?.mul_scalar(2.0)?;

    // Evaluate sync and async
    let sync_result = c1.eval()?;
    let async_result = c2.eval_async().await?;

    // Results should be identical
    assert_eq!(
        sync_result.to_vec1::<f32>()?,
        async_result.to_vec1::<f32>()?
    );

    Ok(())
}

#[tokio::test]
async fn test_async_matmul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let a = LazyTensor::from_tensor(Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
        &device,
    )?)?;
    let b = a.add_tensor_to_graph(Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
        &device,
    )?)?;

    let c = a.matmul(&b)?;

    // Async eval
    let result = c.eval_async().await?;

    // Verify matmul result
    let result_vec = result.to_vec2::<f32>()?;
    assert_eq!(result_vec[0], vec![22.0, 28.0]);
    assert_eq!(result_vec[1], vec![49.0, 64.0]);

    Ok(())
}

#[tokio::test]
async fn test_async_lora_chain() -> Result<(), Box<dyn std::error::Error>> {
    use metal_candle::training::{LoRAConfig, LoRALayer};

    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };

    let lora1 = LoRALayer::new(32, 32, &config, &device)?;
    let lora2 = LoRALayer::new(32, 32, &config, &device)?;

    let input = Tensor::randn(0f32, 1f32, &[4, 32], &device)?;

    // Build lazy graph
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let output1 = lora1.forward_lazy(&input_lazy)?;
    let output2 = lora2.forward_lazy(&output1)?;

    // Async evaluation
    let async_result = output2.eval_async().await?;

    // Compare with sync
    let sync_result = {
        let input_lazy = LazyTensor::from_tensor(input)?;
        let output1 = lora1.forward_lazy(&input_lazy)?;
        let output2 = lora2.forward_lazy(&output1)?;
        output2.eval()?
    };

    let diff = (async_result - sync_result)?.abs()?.flatten_all()?;
    assert!(diff.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[tokio::test]
async fn test_async_softmax() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let input = Tensor::randn(0f32, 1f32, &[4, 8], &device)?;
    let input_lazy = LazyTensor::from_tensor(input)?;

    let output = input_lazy.softmax(1)?;

    // Async eval
    let result = output.eval_async().await?;

    // Verify softmax properties
    assert_eq!(result.dims(), &[4, 8]);

    // Sum along dim should be ~1
    let sum = result.sum(1)?;
    let sum_vec = sum.to_vec1::<f32>()?;
    for s in sum_vec {
        assert!((s - 1.0).abs() < 1e-5);
    }

    Ok(())
}

#[tokio::test]
async fn test_async_rms_norm() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    let input = Tensor::randn(0f32, 1f32, &[4, 64], &device)?;
    let input_lazy = LazyTensor::from_tensor(input)?;

    let output = input_lazy.rms_norm(1e-5)?;

    // Async eval
    let result = output.eval_async().await?;

    // Verify shape
    assert_eq!(result.dims(), &[4, 64]);

    Ok(())
}

#[tokio::test]
async fn test_async_complex_graph() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Build complex graph: (a @ b) + c * 2.0 -> softmax -> rms_norm
    let a = LazyTensor::from_tensor(Tensor::from_slice(&[1.0f32; 12], &[3, 4], &device)?)?;
    let b = a.add_tensor_to_graph(Tensor::from_slice(&[1.0f32; 16], &[4, 4], &device)?)?;
    let c = a.add_tensor_to_graph(Tensor::from_slice(&[0.5f32; 12], &[3, 4], &device)?)?;

    let matmul = a.matmul(&b)?;
    let add = matmul.add(&c)?;
    let scaled = add.mul_scalar(2.0)?;
    let soft = scaled.softmax(1)?;
    let normed = soft.rms_norm(1e-5)?;

    // Async eval entire chain
    let result = normed.eval_async().await?;

    // Verify shape preserved
    assert_eq!(result.dims(), &[3, 4]);

    // Sync eval for comparison
    let sync_result = {
        let a = LazyTensor::from_tensor(Tensor::from_slice(&[1.0f32; 12], &[3, 4], &device)?)?;
        let b = a.add_tensor_to_graph(Tensor::from_slice(&[1.0f32; 16], &[4, 4], &device)?)?;
        let c = a.add_tensor_to_graph(Tensor::from_slice(&[0.5f32; 12], &[3, 4], &device)?)?;

        let matmul = a.matmul(&b)?;
        let add = matmul.add(&c)?;
        let scaled = add.mul_scalar(2.0)?;
        let soft = scaled.softmax(1)?;
        let normed = soft.rms_norm(1e-5)?;
        normed.eval()?
    };

    // Should match
    let diff = (result - sync_result)?.abs()?.flatten_all()?;
    assert!(diff.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}
