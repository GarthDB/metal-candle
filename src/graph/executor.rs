//! Async executor for computation graphs.

use super::Operation;
use crate::error::TrainingError;
use candle_core::{Device, Tensor};

#[cfg(feature = "custom-metal")]
use crate::backend::CustomMetalOps;

/// Executes computation graphs using the underlying backend.
///
/// For now, this is a simple synchronous executor that wraps Candle operations.
/// In Phase 4+, this will be enhanced with:
/// - Async Metal command buffer batching
/// - Multi-operation command buffer encoding
/// - Parallel execution where possible
pub struct AsyncExecutor {
    /// Device for execution
    device: Device,
}

impl AsyncExecutor {
    /// Create a new executor
    ///
    /// # Errors
    ///
    /// Returns error if device initialization fails
    pub fn new(device: Device) -> Result<Self, TrainingError> {
        Ok(Self { device })
    }

    /// Execute a single operation
    ///
    /// # Errors
    ///
    /// Returns error if operation execution fails
    pub fn execute_operation(
        &mut self,
        operation: &Operation,
        inputs: &[Tensor],
    ) -> Result<Tensor, TrainingError> {
        match operation {
            Operation::Input => Err(TrainingError::Failed {
                reason: "Cannot execute Input operation".to_string(),
            }),
            Operation::Matmul => Self::execute_matmul(inputs),
            Operation::Add => Self::execute_add(inputs),
            Operation::Mul => Self::execute_mul(inputs),
            Operation::MulScalar { value } => Self::execute_mul_scalar(inputs, *value),
            #[cfg(feature = "custom-metal")]
            Operation::LoRA { scale, .. } => Self::execute_lora(inputs, *scale),
            #[cfg(feature = "custom-metal")]
            Operation::Softmax { dim } => Self::execute_softmax(inputs, *dim),
            #[cfg(feature = "custom-metal")]
            Operation::RMSNorm { eps } => Self::execute_rmsnorm(inputs, *eps),
        }
    }

    fn execute_matmul(inputs: &[Tensor]) -> Result<Tensor, TrainingError> {
        if inputs.len() != 2 {
            return Err(TrainingError::Failed {
                reason: format!("Matmul requires 2 inputs, got {}", inputs.len()),
            });
        }
        inputs[0]
            .broadcast_matmul(&inputs[1])
            .map_err(|e| TrainingError::Failed {
                reason: format!("Matmul failed: {e}"),
            })
    }

    fn execute_add(inputs: &[Tensor]) -> Result<Tensor, TrainingError> {
        if inputs.len() != 2 {
            return Err(TrainingError::Failed {
                reason: format!("Add requires 2 inputs, got {}", inputs.len()),
            });
        }
        inputs[0]
            .broadcast_add(&inputs[1])
            .map_err(|e| TrainingError::Failed {
                reason: format!("Add failed: {e}"),
            })
    }

    fn execute_mul(inputs: &[Tensor]) -> Result<Tensor, TrainingError> {
        if inputs.len() != 2 {
            return Err(TrainingError::Failed {
                reason: format!("Mul requires 2 inputs, got {}", inputs.len()),
            });
        }
        inputs[0]
            .broadcast_mul(&inputs[1])
            .map_err(|e| TrainingError::Failed {
                reason: format!("Mul failed: {e}"),
            })
    }

    fn execute_mul_scalar(inputs: &[Tensor], value: f32) -> Result<Tensor, TrainingError> {
        if inputs.len() != 1 {
            return Err(TrainingError::Failed {
                reason: format!("MulScalar requires 1 input, got {}", inputs.len()),
            });
        }
        inputs[0]
            .affine(f64::from(value), 0.0)
            .map_err(|e| TrainingError::Failed {
                reason: format!("MulScalar failed: {e}"),
            })
    }

    #[cfg(feature = "custom-metal")]
    fn execute_lora(inputs: &[Tensor], scale: f32) -> Result<Tensor, TrainingError> {
        if inputs.len() != 3 {
            return Err(TrainingError::Failed {
                reason: format!("LoRA requires 3 inputs (input, a, b), got {}", inputs.len()),
            });
        }

        let input = &inputs[0];
        let lora_a = &inputs[1];
        let lora_b = &inputs[2];

        // Try custom fused LoRA kernel
        if input.device().is_metal() {
            if let Ok(output) = input.lora_forward_fused(lora_a, lora_b, scale) {
                return Ok(output);
            }
        }

        // Fallback: sequential operations
        let hidden = input
            .broadcast_matmul(lora_a)
            .map_err(|e| TrainingError::Failed {
                reason: format!("LoRA matmul A failed: {e}"),
            })?;

        let output = hidden
            .broadcast_matmul(lora_b)
            .map_err(|e| TrainingError::Failed {
                reason: format!("LoRA matmul B failed: {e}"),
            })?;

        output
            .affine(f64::from(scale), 0.0)
            .map_err(|e| TrainingError::Failed {
                reason: format!("LoRA scaling failed: {e}"),
            })
    }

    #[cfg(feature = "custom-metal")]
    fn execute_softmax(inputs: &[Tensor], dim: usize) -> Result<Tensor, TrainingError> {
        use candle_nn::ops::softmax;

        if inputs.len() != 1 {
            return Err(TrainingError::Failed {
                reason: format!("Softmax requires 1 input, got {}", inputs.len()),
            });
        }

        // Use Candle's softmax implementation
        // See issue #27 for custom Metal kernel optimization
        softmax(&inputs[0], dim).map_err(|e| TrainingError::Failed {
            reason: format!("Softmax failed: {e}"),
        })
    }

    #[cfg(feature = "custom-metal")]
    fn execute_rmsnorm(inputs: &[Tensor], eps: f32) -> Result<Tensor, TrainingError> {
        if inputs.len() != 1 {
            return Err(TrainingError::Failed {
                reason: format!("RMSNorm requires 1 input, got {}", inputs.len()),
            });
        }

        // Use custom fused RMS norm kernel if available
        if inputs[0].device().is_metal() {
            if let Ok(output) = inputs[0].rms_norm_fused(eps) {
                return Ok(output);
            }
        }

        // Fallback: Use candle_nn's rms_norm (expects alpha of shape [last_dim])
        let input_dims = inputs[0].dims();
        let last_dim = *input_dims.last().unwrap_or(&1);
        let alpha =
            Tensor::ones(&[last_dim], inputs[0].dtype(), inputs[0].device()).map_err(|e| {
                TrainingError::Failed {
                    reason: format!("Failed to create alpha tensor for RMSNorm: {e}"),
                }
            })?;

        candle_nn::ops::rms_norm(&inputs[0], &alpha, eps).map_err(|e| TrainingError::Failed {
            reason: format!("RMSNorm operation failed: {e}"),
        })
    }

    /// Synchronize - wait for all pending operations to complete
    ///
    /// # Errors
    ///
    /// Returns error if synchronization fails
    pub fn synchronize(&mut self) -> Result<(), TrainingError> {
        // For now, Candle operations are synchronous
        // In Phase 4+, this will wait for Metal command buffers to complete
        Ok(())
    }
}

impl std::fmt::Debug for AsyncExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncExecutor")
            .field("device", &self.device)
            .finish()
    }
}
