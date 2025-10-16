//! Training coordinator for `LoRA` fine-tuning.
//!
//! This module provides the `Trainer` struct for coordinating the training process,
//! including forward passes, loss computation, backward passes, and optimizer updates.

use crate::error::Result;
use crate::training::{AdamW, AdamWConfig, LRScheduler, LoRAAdapter, LoRAAdapterConfig};
use candle_core::{Device, Tensor, Var};
use std::time::Instant;

/// Training configuration.
///
/// Configures all aspects of the training process including optimizer, learning rate,
/// and training loop hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub num_epochs: usize,

    /// Learning rate scheduler
    pub lr_scheduler: LRScheduler,

    /// Optimizer configuration
    pub optimizer_config: AdamWConfig,

    /// Maximum gradient norm for clipping (None = no clipping)
    pub max_grad_norm: Option<f32>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 3,
            lr_scheduler: LRScheduler::Constant { lr: 1e-4 },
            optimizer_config: AdamWConfig::default(),
            max_grad_norm: Some(1.0),
        }
    }
}

/// High-level trainer for `LoRA` fine-tuning.
///
/// Coordinates the full training process including epochs, batches, LR scheduling,
/// and progress tracking.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::{Trainer, TrainingConfig, LoRAAdapterConfig};
/// use candle_core::Device;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let lora_config = LoRAAdapterConfig::default();
/// let training_config = TrainingConfig::default();
///
/// // Create trainer
/// let mut trainer = Trainer::new(32, 128, 12, &lora_config, training_config, &device)?;
///
/// // Train (with your model's forward function)
/// // let metrics = trainer.train(&dataset, forward_fn)?;
/// # Ok(())
/// # }
/// ```
pub struct Trainer {
    lora_adapter: LoRAAdapter,
    optimizer: AdamW,
    config: TrainingConfig,
    training_step: TrainingStep,
    global_step: usize,
}

impl Trainer {
    /// Creates a new trainer.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Model hidden dimension
    /// * `intermediate_size` - MLP intermediate dimension
    /// * `num_layers` - Number of transformer layers
    /// * `lora_config` - `LoRA` adapter configuration
    /// * `training_config` - Training hyperparameters
    /// * `device` - Device to place tensors on
    ///
    /// # Errors
    ///
    /// Returns an error if `LoRA` adapter or optimizer initialization fails.
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_layers: usize,
        lora_config: &LoRAAdapterConfig,
        training_config: TrainingConfig,
        device: &Device,
    ) -> Result<Self> {
        let lora_adapter = LoRAAdapter::new(
            hidden_size,
            intermediate_size,
            num_layers,
            lora_config,
            device,
        )?;

        let optimizer = AdamW::new(training_config.optimizer_config)?;

        Ok(Self {
            lora_adapter,
            optimizer,
            config: training_config,
            training_step: TrainingStep::new(),
            global_step: 0,
        })
    }

    /// Trains the model for the configured number of epochs.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Iterator over (`input_ids`, `target_ids`) batches
    /// * `forward_fn` - Model forward pass function
    ///
    /// # Returns
    ///
    /// Vector of `StepMetrics` for each training step.
    ///
    /// # Errors
    ///
    /// Returns an error if any training step fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use metal_candle::training::Trainer;
    /// use candle_core::Tensor;
    ///
/// # fn example(mut trainer: Trainer) -> Result<(), Box<dyn std::error::Error>> {
/// # use candle_core::{Tensor, Device, DType};
/// // Prepare dataset (batches of input/target tensors)
/// let device = Device::Cpu;
/// let input = Tensor::zeros((1, 8), DType::U32, &device)?;
/// let target = Tensor::zeros((1, 8), DType::U32, &device)?;
/// let dataset = vec![(input, target)];
///
/// // Define forward pass (returns metal_candle::Result)
/// let forward_fn = |_input: &Tensor| -> metal_candle::Result<Tensor> {
///     Ok(Tensor::zeros((1, 8, 100), DType::F32, &device)?)
/// };
///
/// // Train
/// let metrics = trainer.train(&dataset, forward_fn)?;
/// # Ok(())
/// # }
    /// ```
    pub fn train<F, E>(
        &mut self,
        dataset: &[(Tensor, Tensor)],
        forward_fn: F,
    ) -> Result<Vec<StepMetrics>>
    where
        F: Fn(&Tensor) -> std::result::Result<Tensor, E>,
        E: Into<crate::error::Error>,
    {
        let mut all_metrics = Vec::new();
        let total_steps = self.config.num_epochs * dataset.len();

        println!("\n🚀 Starting LoRA Training");
        println!("═══════════════════════════");
        println!("Epochs: {}", self.config.num_epochs);
        println!("Batches per epoch: {}", dataset.len());
        println!("Total steps: {total_steps}");
        let trainable_params = self.lora_adapter.num_trainable_parameters();
        println!("LoRA trainable params: {trainable_params}");
        println!();

        let training_start = Instant::now();

        for epoch in 0..self.config.num_epochs {
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;

            println!("Epoch {}/{}", epoch + 1, self.config.num_epochs);
            println!("─────────────────────────");

            for (batch_idx, (input_ids, target_ids)) in dataset.iter().enumerate() {
                // Get current learning rate from scheduler
                let lr = self.config.lr_scheduler.get_lr(self.global_step);

                // Wrap forward_fn to convert error type
                let forward_wrapper =
                    |input: &Tensor| -> Result<Tensor> { forward_fn(input).map_err(Into::into) };

                // Execute training step
                let metrics = self.training_step.execute(
                    input_ids,
                    target_ids,
                    &self.lora_adapter,
                    &mut self.optimizer,
                    lr,
                    forward_wrapper,
                )?;

                epoch_loss += metrics.loss;
                all_metrics.push(metrics.clone());
                self.global_step += 1;

                // Print progress every 10 steps
                if (batch_idx + 1) % 10 == 0 || batch_idx + 1 == dataset.len() {
                    println!(
                        "  Step {}/{} | Loss: {:.4} | LR: {:.6}",
                        batch_idx + 1,
                        dataset.len(),
                        metrics.loss,
                        metrics.learning_rate
                    );
                }
            }

            let epoch_time = epoch_start.elapsed();
            #[allow(clippy::cast_precision_loss)]
            let avg_epoch_loss = epoch_loss / dataset.len() as f32; // Safe: batch count is reasonable

            println!(
                "  Epoch {} complete | Avg Loss: {:.4} | Time: {:.2}s",
                epoch + 1,
                avg_epoch_loss,
                epoch_time.as_secs_f32()
            );
            println!();
        }

        let total_time = training_start.elapsed();
        println!("✅ Training Complete!");
        println!("═══════════════════════");
        println!("Total time: {:.2}s", total_time.as_secs_f32());
        let final_loss = all_metrics.last().map_or(0.0, |m| m.loss);
        println!("Final loss: {final_loss:.4}");
        println!();

        Ok(all_metrics)
    }

    /// Returns a reference to the `LoRA` adapter.
    #[must_use]
    pub const fn lora_adapter(&self) -> &LoRAAdapter {
        &self.lora_adapter
    }

    /// Returns a mutable reference to the `LoRA` adapter.
    #[must_use]
    pub fn lora_adapter_mut(&mut self) -> &mut LoRAAdapter {
        &mut self.lora_adapter
    }

    /// Returns the global step count.
    #[must_use]
    pub const fn global_step(&self) -> usize {
        self.global_step
    }
}

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
