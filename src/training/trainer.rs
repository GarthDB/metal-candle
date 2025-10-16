//! Training coordinator for `LoRA` fine-tuning.
//!
//! This module provides the `Trainer` struct for coordinating the training process,
//! including forward passes, loss computation, backward passes, and optimizer updates.

use crate::error::Result;
use crate::training::{AdamW, LoRAAdapter};
use candle_core::{Tensor, Var};

/// Training step result.
///
/// Contains metrics from a single training step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    /// Training loss
    pub loss: f32,

    /// Step number
    pub step: usize,

    /// Learning rate used for this step
    pub learning_rate: f32,
}

/// Single training step coordinator.
///
/// Executes a single forward→loss→backward→update cycle.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::TrainingStep;
/// use candle_core::{Tensor, Device, DType};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
///
/// // Create training step
/// let step = TrainingStep::new();
///
/// // Execute step with your data
/// // let metrics = step.execute(...)?;
/// # Ok(())
/// # }
/// ```
pub struct TrainingStep {
    /// Current step number
    step: usize,
}

impl TrainingStep {
    /// Creates a new training step coordinator.
    #[must_use]
    pub fn new() -> Self {
        Self { step: 0 }
    }

    /// Executes a single training step.
    ///
    /// # Process
    ///
    /// 1. **Forward pass**: Compute model predictions
    /// 2. **Loss computation**: Calculate training loss
    /// 3. **Backward pass**: Compute gradients via autograd
    /// 4. **Optimizer update**: Update trainable parameters
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs (batch, `sequence_length`)
    /// * `target_ids` - Target token IDs (batch, `sequence_length`)
    /// * `lora_adapter` - `LoRA` adapter with trainable parameters
    /// * `optimizer` - Optimizer for parameter updates
    /// * `learning_rate` - Current learning rate
    /// * `forward_fn` - Forward pass function (takes `input_ids`, returns logits)
    ///
    /// # Returns
    ///
    /// `StepMetrics` containing loss and step information.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Forward pass fails
    /// - Loss computation fails
    /// - Backward pass fails
    /// - Optimizer update fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::training::{TrainingStep, AdamW, LoRAAdapter};
    /// use candle_core::{Tensor, Device, DType};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let mut step = TrainingStep::new();
    ///
    /// // Prepare data
    /// let input_ids = Tensor::zeros((2, 128), DType::U32, &device)?;
    /// let target_ids = Tensor::zeros((2, 128), DType::U32, &device)?;
    ///
    /// // Setup LoRA and optimizer (simplified)
    /// // let lora_adapter = ...;
    /// // let mut optimizer = AdamW::new(...);
    ///
    /// // Define forward pass
    /// let forward_fn = |input: &Tensor| -> Result<Tensor, Box<dyn std::error::Error>> {
    ///     // Your model forward pass here
    ///     Ok(Tensor::zeros((2, 128, 32000), DType::F32, &device)?)
    /// };
    ///
    /// // Execute training step
    /// // let metrics = step.execute(&input_ids, &target_ids, &lora_adapter, &mut optimizer, 1e-4, forward_fn)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute<F>(
        &mut self,
        input_ids: &Tensor,
        target_ids: &Tensor,
        lora_adapter: &LoRAAdapter,
        optimizer: &mut AdamW,
        learning_rate: f32,
        forward_fn: F,
    ) -> Result<StepMetrics>
    where
        F: Fn(&Tensor) -> Result<Tensor>,
    {
        // Step 1: Forward pass
        let logits = forward_fn(input_ids)?;

        // Step 2: Loss computation
        // Use -100 as ignore index (common convention for padding tokens)
        let loss = crate::training::cross_entropy_loss(&logits, target_ids, Some(u32::MAX))?;

        // Step 3: Backward pass (compute gradients)
        let grads = loss.backward()?;

        // Step 4: Optimizer update
        // Collect all trainable variables from LoRA adapter
        let trainable_vars = Self::collect_trainable_vars(lora_adapter);

        // Update learning rate in optimizer
        optimizer.set_lr(learning_rate);

        // Update each parameter using its gradient
        for var in trainable_vars {
            if let Some(grad) = grads.get(var) {
                optimizer.step_var(var, grad)?;
            }
        }

        // Extract loss value for metrics
        let loss_value = loss.to_vec0::<f32>()?;

        self.step += 1;

        Ok(StepMetrics {
            loss: loss_value,
            step: self.step,
            learning_rate,
        })
    }

    /// Collects all trainable variables from the `LoRA` adapter.
    ///
    /// This helper method gathers all `Var` parameters that need gradients
    /// from the adapter's `LoRA` layers.
    fn collect_trainable_vars(lora_adapter: &LoRAAdapter) -> Vec<&Var> {
        let mut vars = Vec::new();

        // Iterate through all LoRA layers in the adapter
        // layers() returns an iterator over (&String, &LoRALayer)
        for (_key, layer) in lora_adapter.layers() {
            vars.extend(layer.trainable_variables());
        }

        vars
    }

    /// Returns the current step number.
    #[must_use]
    pub const fn step(&self) -> usize {
        self.step
    }

    /// Resets the step counter.
    pub fn reset(&mut self) {
        self.step = 0;
    }
}

impl Default for TrainingStep {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{AdamWConfig, LoRAAdapterConfig, TargetModule};
    use candle_core::{DType, Device};

    #[test]
    fn test_training_step_creation() {
        let step = TrainingStep::new();
        assert_eq!(step.step(), 0);
    }

    #[test]
    fn test_training_step_reset() {
        let mut step = TrainingStep::new();
        step.step = 5;
        step.reset();
        assert_eq!(step.step(), 0);
    }

    #[test]
    fn test_training_step_execute() {
        let device = Device::Cpu;

        // Create LoRA adapter
        let adapter_config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        // Create LoRA adapter with proper dimensions
        let hidden_size = 32;
        let intermediate_size = 128;
        let num_layers = 1;

        let lora_adapter = LoRAAdapter::new(
            hidden_size,
            intermediate_size,
            num_layers,
            &adapter_config,
            &device,
        )
        .unwrap();

        // Create optimizer
        let opt_config = AdamWConfig::default();
        let mut optimizer = AdamW::new(opt_config).unwrap();

        // Create training step
        let mut step = TrainingStep::new();

        // Prepare dummy data
        let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
        let target_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();

        // Define a simple forward function that returns logits
        let forward_fn = |_input: &Tensor| -> Result<Tensor> {
            // Return dummy logits: (batch=1, seq_len=8, vocab_size=100)
            Ok(Tensor::zeros((1, 8, 100), DType::F32, &device)?)
        };

        // Execute step
        let metrics = step.execute(
            &input_ids,
            &target_ids,
            &lora_adapter,
            &mut optimizer,
            1e-4,
            forward_fn,
        );

        assert!(metrics.is_ok(), "Training step should succeed");
        let metrics = metrics.unwrap();
        assert_eq!(metrics.step, 1);
        assert_eq!(metrics.learning_rate, 1e-4);
        assert!(metrics.loss >= 0.0, "Loss should be non-negative");
    }
}
