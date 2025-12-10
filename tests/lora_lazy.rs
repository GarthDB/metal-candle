//! Tests for LoRA lazy execution.

#![cfg(feature = "graph")]

use candle_core::{Device, Tensor};
use metal_candle::graph::LazyTensor;
use metal_candle::training::{LoRAConfig, LoRALayer};

#[test]
fn test_lora_lazy_basic() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create LoRA layer
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };
    let lora = LoRALayer::new(16, 16, &config, &device)?;

    // Create input
    let input_data: Vec<f32> = (0..16).map(|i| i as f32 / 16.0).collect();
    let input = Tensor::from_slice(&input_data, &[1, 16], &device)?;

    // Eager execution
    let eager_output = lora.forward(&input)?;

    // Lazy execution
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let output_lazy = lora.forward_lazy(&input_lazy)?;
    let lazy_output = output_lazy.eval()?;

    // Validate shapes match
    assert_eq!(eager_output.shape(), lazy_output.shape());

    // Validate results are close
    let diff = (eager_output - lazy_output)?.abs()?;
    // Flatten to 1D and get max
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
fn test_lora_lazy_batched() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create LoRA layer
    let config = LoRAConfig::default();
    let lora = LoRALayer::new(64, 64, &config, &device)?;

    // Batched input
    let input = Tensor::randn(0f32, 1f32, &[8, 64], &device)?;

    // Eager
    let eager_output = lora.forward(&input)?;

    // Lazy
    let input_lazy = LazyTensor::from_tensor(input)?;
    let lazy_output = lora.forward_lazy(&input_lazy)?.eval()?;

    // Validate
    assert_eq!(eager_output.dims(), &[8, 64]);
    assert_eq!(lazy_output.dims(), &[8, 64]);

    let diff = (eager_output - lazy_output)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_lora_lazy_chain() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create two LoRA layers
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };
    let lora1 = LoRALayer::new(32, 32, &config, &device)?;
    let lora2 = LoRALayer::new(32, 32, &config, &device)?;

    let input = Tensor::randn(0f32, 1f32, &[4, 32], &device)?;

    // Chain operations in lazy mode
    let input_lazy = LazyTensor::from_tensor(input.clone())?;
    let output1 = lora1.forward_lazy(&input_lazy)?;
    let output2 = lora2.forward_lazy(&output1)?;

    // Single eval executes both
    let lazy_result = output2.eval()?;

    // Compare with eager
    let eager1 = lora1.forward(&input)?;
    let eager2 = lora2.forward(&eager1)?;

    let diff = (lazy_result - eager2)?.abs()?;
    let diff_flat = diff.flatten_all()?;
    assert!(diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4);

    Ok(())
}

#[test]
fn test_lora_lazy_different_ranks() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    for rank in [2, 4, 8, 16] {
        let config = LoRAConfig {
            rank,
            alpha: rank as f32 * 2.0,
            dropout: 0.0,
        };
        let lora = LoRALayer::new(64, 64, &config, &device)?;
        let input = Tensor::randn(0f32, 1f32, &[1, 64], &device)?;

        let eager_output = lora.forward(&input)?;
        let lazy_output = lora
            .forward_lazy(&LazyTensor::from_tensor(input)?)?
            .eval()?;

        let diff = (eager_output - lazy_output)?.abs()?;
        let diff_flat = diff.flatten_all()?;
        assert!(
            diff_flat.max(0)?.to_scalar::<f32>()? < 1e-4,
            "Failed for rank {}",
            rank
        );
    }

    Ok(())
}

#[test]
fn test_lora_lazy_shape_preservation() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let config = LoRAConfig::default();
    let lora = LoRALayer::new(128, 128, &config, &device)?;

    // Test various input shapes
    for (batch, seq_len) in [(1, 10), (4, 20), (8, 5)] {
        let input = Tensor::randn(0f32, 1f32, &[batch, seq_len, 128], &device)?;
        let input_lazy = LazyTensor::from_tensor(input.clone())?;

        let output = lora.forward_lazy(&input_lazy)?;

        // Shape should be known before eval
        assert_eq!(output.shape().dims(), &[batch, seq_len, 128]);

        // Eval should preserve shape
        let result = output.eval()?;
        assert_eq!(result.dims(), &[batch, seq_len, 128]);
    }

    Ok(())
}
