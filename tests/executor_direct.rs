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
