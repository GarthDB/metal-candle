//! Optimizers for training.
//!
//! Implements `AdamW` and other optimization algorithms for updating
//! model parameters during training.

use crate::error::Result;
use candle_core::{Tensor, Var};
use std::collections::HashMap;

/// Configuration for the `AdamW` optimizer.
///
/// `AdamW` is a variant of Adam with decoupled weight decay, which is
/// the standard optimizer for training transformer models.
///
/// # References
///
/// - "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2017)
/// - <https://arxiv.org/abs/1711.05101>
///
/// # Examples
///
/// ```
/// use metal_candle::training::AdamWConfig;
///
/// let config = AdamWConfig {
///     learning_rate: 1e-4,
///     beta1: 0.9,
///     beta2: 0.999,
///     epsilon: 1e-8,
///     weight_decay: 0.01,
/// };
/// ```
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// Learning rate (typically 1e-4 to 1e-3 for transformers)
    pub learning_rate: f32,

    /// Exponential decay rate for first moment estimates (typically 0.9)
    pub beta1: f32,

    /// Exponential decay rate for second moment estimates (typically 0.999)
    pub beta2: f32,

    /// Small constant for numerical stability (typically 1e-8)
    pub epsilon: f32,

    /// Weight decay coefficient (typically 0.01 for `AdamW`)
    pub weight_decay: f32,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }
}

impl AdamWConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of valid range.
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "learning rate must be positive".to_string(),
            }
            .into());
        }

        if !(0.0..1.0).contains(&self.beta1) {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: format!("beta1 must be in [0, 1), got {}", self.beta1),
            }
            .into());
        }

        if !(0.0..1.0).contains(&self.beta2) {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: format!("beta2 must be in [0, 1), got {}", self.beta2),
            }
            .into());
        }

        if self.epsilon <= 0.0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "epsilon must be positive".to_string(),
            }
            .into());
        }

        if self.weight_decay < 0.0 {
            return Err(crate::error::TrainingError::InvalidConfig {
                reason: "weight decay must be non-negative".to_string(),
            }
            .into());
        }

        Ok(())
    }
}

/// Optimizer state for a single parameter.
///
/// Stores the first and second moment estimates for Adam.
#[derive(Debug)]
struct ParameterState {
    /// First moment estimate (exponential moving average of gradients)
    m: Tensor,

    /// Second moment estimate (exponential moving average of squared gradients)
    v: Tensor,
}

impl ParameterState {
    /// Creates a new parameter state initialized to zeros.
    fn new(shape: &[usize], device: &candle_core::Device) -> Result<Self> {
        let m = Tensor::zeros(shape, candle_core::DType::F32, device)?;
        let v = Tensor::zeros(shape, candle_core::DType::F32, device)?;

        Ok(Self { m, v })
    }
}

/// `AdamW` optimizer.
///
/// Implements the `AdamW` algorithm with bias correction and decoupled weight decay.
///
/// # Algorithm
///
/// ```text
/// m_t = β1 * m_{t-1} + (1 - β1) * g_t           (first moment)
/// v_t = β2 * v_{t-1} + (1 - β2) * g_t^2         (second moment)
///
/// m_hat = m_t / (1 - β1^t)                      (bias correction)
/// v_hat = v_t / (1 - β2^t)
///
/// θ_t = θ_{t-1} - lr * (m_hat / (√v_hat + ε) + λ * θ_{t-1})
/// ```
///
/// where λ is the weight decay coefficient.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::{AdamW, AdamWConfig};
/// use candle_core::{Tensor, Device};
/// use std::collections::HashMap;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let device = Device::Cpu;
/// let config = AdamWConfig::default();
/// let mut optimizer = AdamW::new(config)?;
///
/// // Create a parameter
/// let param = Tensor::randn(0f32, 1f32, (100, 100), &device)?;
/// optimizer.add_parameter("weight", &param)?;
///
/// // In training loop:
/// // let grad = compute_gradient(&param)?;
/// // optimizer.step("weight", &param, &grad)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AdamW {
    config: AdamWConfig,
    state: HashMap<String, ParameterState>,
    step_count: usize,
}

impl AdamW {
    /// Creates a new `AdamW` optimizer.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn new(config: AdamWConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            state: HashMap::new(),
            step_count: 0,
        })
    }

    /// Adds a parameter to the optimizer.
    ///
    /// This initializes the optimizer state (m, v) for the parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if state initialization fails.
    pub fn add_parameter(&mut self, name: impl Into<String>, param: &Tensor) -> Result<()> {
        let name = name.into();
        let shape = param.dims();
        let device = param.device();

        let state = ParameterState::new(shape, device)?;
        self.state.insert(name, state);

        Ok(())
    }

    /// Performs a single optimization step for a parameter.
    ///
    /// # Arguments
    ///
    /// * `name` - Parameter name (must have been added via `add_parameter`)
    /// * `param` - Current parameter value
    /// * `grad` - Gradient with respect to the parameter
    ///
    /// # Returns
    ///
    /// The updated parameter value.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameter is not found or tensor operations fail.
    pub fn step(&mut self, name: &str, param: &Tensor, grad: &Tensor) -> Result<Tensor> {
        // Get or create parameter state
        let state =
            self.state
                .get_mut(name)
                .ok_or_else(|| crate::error::TrainingError::StateError {
                    reason: format!("parameter '{name}' not found in optimizer"),
                })?;

        // Increment global step count
        self.step_count += 1;
        let t = self.step_count;

        let config = &self.config;

        // Update first moment: m_t = β1 * m_{t-1} + (1 - β1) * g_t
        let m_new = ((state.m.clone() * f64::from(config.beta1))?
            + (grad * f64::from(1.0 - config.beta1))?)?;

        // Update second moment: v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
        let grad_sq = grad.sqr()?;
        let v_new = ((state.v.clone() * f64::from(config.beta2))?
            + (grad_sq * f64::from(1.0 - config.beta2))?)?;

        // Bias correction
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let t_i32 = t as i32; // Safe: step count unlikely to exceed i32::MAX
        let beta1_t = config.beta1.powi(t_i32);
        let beta2_t = config.beta2.powi(t_i32);

        let m_hat = (m_new.clone() / f64::from(1.0 - beta1_t))?;
        let v_hat = (v_new.clone() / f64::from(1.0 - beta2_t))?;

        // Compute update: m_hat / (√v_hat + ε)
        let v_sqrt = v_hat.sqrt()?;
        let denom = (v_sqrt + f64::from(config.epsilon))?;
        let update = m_hat.div(&denom)?;

        // Apply learning rate
        let lr_update = (update * f64::from(config.learning_rate))?;

        // Apply weight decay (decoupled): λ * θ
        let wd_update = (param * f64::from(config.weight_decay * config.learning_rate))?;

        // Update parameter: θ_t = θ_{t-1} - lr * update - wd * θ
        let param_new = ((param - &lr_update)? - &wd_update)?;

        // Store updated state
        state.m = m_new;
        state.v = v_new;

        Ok(param_new)
    }

    /// Performs a single optimization step for a `Var` parameter (autograd-based).
    ///
    /// This method works with Candle's autograd `Var` type instead of requiring
    /// parameter names. It updates the parameter in-place using its gradient.
    ///
    /// # Arguments
    ///
    /// * `var` - The trainable parameter (Var)
    /// * `grad` - Gradient with respect to the parameter
    ///
    /// # Errors
    ///
    /// Returns an error if tensor operations fail or if the parameter doesn't have state.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    pub fn step_var(&mut self, var: &Var, grad: &Tensor) -> Result<()> {
        // Use the variable's ID as the key
        let var_id = format!("{var:p}");

        // Get or create parameter state
        if !self.state.contains_key(&var_id) {
            let param = var.as_tensor();
            self.add_parameter(&var_id, param)?;
        }

        // Perform update using the named parameter API
        let param = var.as_tensor();
        let updated_param = self.step(&var_id, param, grad)?;

        // Update the Var with the new value
        var.set(&updated_param)?;

        Ok(())
    }

    /// Sets the learning rate for the optimizer.
    ///
    /// This updates the learning rate used for parameter updates.
    pub fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Returns the current learning rate.
    #[must_use]
    pub const fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }
    /// Returns the current step count.
    #[must_use]
    pub const fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns the optimizer configuration.
    #[must_use]
    pub const fn config(&self) -> &AdamWConfig {
        &self.config
    }

    /// Resets the optimizer state (but keeps parameters registered).
    pub fn reset_state(&mut self) {
        self.step_count = 0;
        for state in self.state.values_mut() {
            // Zero out m and v
            if let Ok(zeros_m) =
                Tensor::zeros(state.m.dims(), candle_core::DType::F32, state.m.device())
            {
                state.m = zeros_m;
            }
            if let Ok(zeros_v) =
                Tensor::zeros(state.v.dims(), candle_core::DType::F32, state.v.device())
            {
                state.v = zeros_v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_adamw_config_default() {
        let config = AdamWConfig::default();
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.epsilon, 1e-8);
        assert_eq!(config.weight_decay, 0.01);
    }

    #[test]
    fn test_adamw_config_validation() {
        let valid = AdamWConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_lr = AdamWConfig {
            learning_rate: -1.0,
            ..Default::default()
        };
        assert!(invalid_lr.validate().is_err());

        let invalid_beta1 = AdamWConfig {
            beta1: 1.5,
            ..Default::default()
        };
        assert!(invalid_beta1.validate().is_err());

        let invalid_beta2 = AdamWConfig {
            beta2: -0.1,
            ..Default::default()
        };
        assert!(invalid_beta2.validate().is_err());
    }

    #[test]
    fn test_adamw_creation() {
        let config = AdamWConfig::default();
        let optimizer = AdamW::new(config);
        assert!(optimizer.is_ok());

        let optimizer = optimizer.unwrap();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adamw_add_parameter() {
        let device = Device::Cpu;
        let config = AdamWConfig::default();
        let mut optimizer = AdamW::new(config).unwrap();

        let param = Tensor::zeros((10, 10), candle_core::DType::F32, &device).unwrap();
        let result = optimizer.add_parameter("test_param", &param);
        assert!(result.is_ok());
    }

    #[test]
    fn test_adamw_step() {
        let device = Device::Cpu;
        let config = AdamWConfig {
            learning_rate: 0.1,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(config).unwrap();

        // Create a parameter
        let param = Tensor::ones((5, 5), candle_core::DType::F32, &device).unwrap();
        optimizer.add_parameter("param", &param).unwrap();

        // Create a gradient (all ones)
        let grad = Tensor::ones((5, 5), candle_core::DType::F32, &device).unwrap();

        // Perform a step
        let param_new = optimizer.step("param", &param, &grad);
        assert!(param_new.is_ok());

        let param_new = param_new.unwrap();
        assert_eq!(param_new.dims(), &[5, 5]);

        // Parameter should have changed
        let param_sum = param.sum_all().unwrap().to_scalar::<f32>().unwrap();
        let param_new_sum = param_new.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert_ne!(param_sum, param_new_sum);

        // Step count should have incremented
        assert_eq!(optimizer.step_count(), 1);
    }

    #[test]
    fn test_adamw_multiple_steps() {
        let device = Device::Cpu;
        let config = AdamWConfig::default();
        let mut optimizer = AdamW::new(config).unwrap();

        let param = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();
        optimizer.add_parameter("param", &param).unwrap();

        let grad = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();

        // Perform multiple steps
        let mut current_param = param;
        for _ in 0..5 {
            current_param = optimizer.step("param", &current_param, &grad).unwrap();
        }

        assert_eq!(optimizer.step_count(), 5);

        // After 5 steps with constant gradient of 1.0, param should have decreased
        let final_sum = current_param.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(final_sum < 9.0, "final_sum = {}", final_sum); // Started at 9.0 (3x3 ones)
    }

    #[test]
    fn test_adamw_reset_state() {
        let device = Device::Cpu;
        let config = AdamWConfig::default();
        let mut optimizer = AdamW::new(config).unwrap();

        let param = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();
        optimizer.add_parameter("param", &param).unwrap();

        let grad = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();

        // Perform some steps
        optimizer.step("param", &param, &grad).unwrap();
        optimizer.step("param", &param, &grad).unwrap();
        assert_eq!(optimizer.step_count(), 2);

        // Reset state
        optimizer.reset_state();
        assert_eq!(optimizer.step_count(), 0);
    }

    #[test]
    fn test_adamw_parameter_not_found() {
        let device = Device::Cpu;
        let config = AdamWConfig::default();
        let mut optimizer = AdamW::new(config).unwrap();

        let param = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();
        let grad = Tensor::ones((3, 3), candle_core::DType::F32, &device).unwrap();

        // Try to step without adding parameter
        let result = optimizer.step("unknown", &param, &grad);
        assert!(result.is_err());
    }
}
