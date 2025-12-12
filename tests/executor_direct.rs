//! Direct tests for the graph executor to improve coverage.
//!
//! These tests directly exercise the executor's error handling and validation logic.

#![cfg(feature = "graph")]

use candle_core::{Device, Tensor};
use metal_candle::graph::{AsyncExecutor, NodeId, Operation};

#[test]
fn test_executor_creation() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device)?;
    assert!(executor.synchronize().is_ok());
    Ok(())
}

#[test]
fn test_executor_input_operation_fails() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let result = executor.execute_operation(&Operation::Input, &[input]);

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Cannot execute Input operation"));
    Ok(())
}

#[test]
fn test_executor_matmul_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 1 input (needs 2)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let result = executor.execute_operation(&Operation::Matmul, &[a]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Matmul requires 2 inputs"));

    // Test with 3 inputs (needs 2)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2, 1], &device)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[1, 2], &device)?;
    let c = Tensor::from_slice(&[5.0f32, 6.0], &[2], &device)?;
    let result = executor.execute_operation(&Operation::Matmul, &[a, b, c]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Matmul requires 2 inputs"));

    Ok(())
}

#[test]
fn test_executor_add_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 1 input (needs 2)
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let result = executor.execute_operation(&Operation::Add, &[a]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Add requires 2 inputs"));

    Ok(())
}

#[test]
fn test_executor_mul_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 1 input (needs 2)
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let result = executor.execute_operation(&Operation::Mul, &[a]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Mul requires 2 inputs"));

    Ok(())
}

#[test]
fn test_executor_mul_scalar_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 0 inputs (needs 1)
    let result = executor.execute_operation(&Operation::MulScalar { value: 2.0 }, &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("MulScalar requires 1 input"));

    // Test with 2 inputs (needs 1)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2], &device)?;
    let result = executor.execute_operation(&Operation::MulScalar { value: 2.0 }, &[a, b]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("MulScalar requires 1 input"));

    Ok(())
}

#[test]
fn test_executor_matmul_success() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device)?;

    let result = executor.execute_operation(&Operation::Matmul, &[a, b])?;
    assert_eq!(result.dims(), &[2, 2]);

    Ok(())
}

#[test]
fn test_executor_add_success() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device)?;

    let result = executor.execute_operation(&Operation::Add, &[a, b])?;
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1::<f32>()?, vec![5.0, 7.0, 9.0]);

    Ok(())
}

#[test]
fn test_executor_mul_success() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let b = Tensor::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device)?;

    let result = executor.execute_operation(&Operation::Mul, &[a, b])?;
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1::<f32>()?, vec![2.0, 6.0, 12.0]);

    Ok(())
}

#[test]
fn test_executor_mul_scalar_success() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;

    let result = executor.execute_operation(&Operation::MulScalar { value: 2.5 }, &[a])?;
    assert_eq!(result.dims(), &[3]);
    assert_eq!(result.to_vec1::<f32>()?, vec![2.5, 5.0, 7.5]);

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_softmax_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 0 inputs (needs 1)
    let result = executor.execute_operation(&Operation::Softmax { dim: 0 }, &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Softmax requires 1 input"));

    // Test with 2 inputs (needs 1)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2], &device)?;
    let result = executor.execute_operation(&Operation::Softmax { dim: 0 }, &[a, b]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Softmax requires 1 input"));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_rmsnorm_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 0 inputs (needs 1)
    let result = executor.execute_operation(&Operation::RMSNorm { eps: 1e-5 }, &[]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("RMSNorm requires 1 input"));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_lora_wrong_input_count() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 2 inputs (needs 3)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2], &device)?;
    let result = executor.execute_operation(
        &Operation::LoRA {
            a: NodeId(0),
            b: NodeId(1),
            scale: 1.0,
        },
        &[a, b],
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("LoRA requires 3 inputs"));

    Ok(())
}

#[test]
fn test_executor_broadcast_add() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Broadcasting: [3, 1] + [3] -> [3, 3]
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device)?;
    let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0], &[3], &device)?;

    let result = executor.execute_operation(&Operation::Add, &[a, b])?;
    assert_eq!(result.dims(), &[3, 3]);

    Ok(())
}

#[test]
fn test_executor_broadcast_mul() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Broadcasting: [2, 1] * [2] -> [2, 2]
    let a = Tensor::from_slice(&[2.0f32, 3.0], &[2, 1], &device)?;
    let b = Tensor::from_slice(&[4.0f32, 5.0], &[2], &device)?;

    let result = executor.execute_operation(&Operation::Mul, &[a, b])?;
    assert_eq!(result.dims(), &[2, 2]);

    Ok(())
}

#[test]
fn test_executor_synchronize_no_op() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device)?;

    // Multiple synchronize calls should be fine
    executor.synchronize()?;
    executor.synchronize()?;
    executor.synchronize()?;

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_lora_more_input_counts() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 1 input (needs 3)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let result = executor.execute_operation(
        &Operation::LoRA {
            a: NodeId(0),
            b: NodeId(1),
            scale: 1.0,
        },
        &[a],
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("LoRA requires 3 inputs"));

    // Test with 4 inputs (needs 3)
    let a = Tensor::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], &[2], &device)?;
    let c = Tensor::from_slice(&[5.0f32, 6.0], &[2], &device)?;
    let d = Tensor::from_slice(&[7.0f32, 8.0], &[2], &device)?;
    let result = executor.execute_operation(
        &Operation::LoRA {
            a: NodeId(0),
            b: NodeId(1),
            scale: 1.0,
        },
        &[a, b, c, d],
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("LoRA requires 3 inputs"));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_lora_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Create test tensors: input (2x4), lora_a (4x2), lora_b (2x4)
    // Result should be (2x4)
    let input = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    )?;
    let lora_a = Tensor::from_slice(
        &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        &[4, 2],
        &device,
    )?;
    let lora_b = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    )?;

    let result = executor.execute_operation(
        &Operation::LoRA {
            a: NodeId(0),
            b: NodeId(1),
            scale: 0.5,
        },
        &[input, lora_a, lora_b],
    )?;

    assert_eq!(result.dims(), &[2, 4]);
    // Verify it's not all zeros
    let values = result.to_vec2::<f32>()?;
    assert!(values.iter().flatten().any(|&x| x != 0.0));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_lora_metal_success() -> Result<(), Box<dyn std::error::Error>> {
    // Test LoRA on Metal device if available
    let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
        return Ok(());
    };
    {
        let mut executor = AsyncExecutor::new(device.clone())?;

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device)?;
        let lora_a = Tensor::from_slice(
            &[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            &[4, 2],
            &device,
        )?;
        let lora_b = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        )?;

        let result = executor.execute_operation(
            &Operation::LoRA {
                a: NodeId(0),
                b: NodeId(1),
                scale: 1.0,
            },
            &[input, lora_a, lora_b],
        )?;

        assert_eq!(result.dims(), &[1, 4]);
    }

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_rmsnorm_more_input_counts() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with 2 inputs (needs 1)
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device)?;
    let result = executor.execute_operation(&Operation::RMSNorm { eps: 1e-5 }, &[a, b]);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("RMSNorm requires 1 input"));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_rmsnorm_cpu_fallback() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device)?;

    let result = executor.execute_operation(&Operation::RMSNorm { eps: 1e-6 }, &[input])?;

    assert_eq!(result.dims(), &[4]);
    // RMS norm should preserve shape
    let values = result.to_vec1::<f32>()?;
    assert_eq!(values.len(), 4);
    // Values should be normalized
    assert!(values.iter().all(|&x| x.is_finite()));

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_rmsnorm_metal_success() -> Result<(), Box<dyn std::error::Error>> {
    // Test RMSNorm on Metal device if available
    let Ok(Ok(device)) = std::panic::catch_unwind(|| Device::new_metal(0)) else {
        return Ok(());
    };
    {
        let mut executor = AsyncExecutor::new(device.clone())?;

        let input = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        )?;

        let result = executor.execute_operation(&Operation::RMSNorm { eps: 1e-5 }, &[input])?;

        assert_eq!(result.dims(), &[2, 4]);
    }

    Ok(())
}

#[test]
#[cfg(feature = "custom-metal")]
fn test_executor_rmsnorm_correctness() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let mut executor = AsyncExecutor::new(device.clone())?;

    // Test with known values
    let input = Tensor::from_slice(&[3.0f32, 4.0], &[2], &device)?;

    let result = executor.execute_operation(&Operation::RMSNorm { eps: 0.0 }, &[input])?;

    // RMS of [3, 4] is sqrt((9+16)/2) = sqrt(12.5) = 3.5355...
    // Normalized: [3/3.5355, 4/3.5355] ≈ [0.8485, 1.1314]
    let values = result.to_vec1::<f32>()?;
    assert!((values[0] - 0.8485).abs() < 0.001);
    assert!((values[1] - 1.1314).abs() < 0.001);

    Ok(())
}

#[test]
fn test_executor_debug_impl() {
    let device = Device::Cpu;
    let executor = AsyncExecutor::new(device).unwrap();

    let debug_string = format!("{:?}", executor);
    assert!(debug_string.contains("AsyncExecutor"));
    assert!(debug_string.contains("device"));
}
