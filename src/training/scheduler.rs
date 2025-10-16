//! Learning rate schedulers for training.
//!
//! Implements various learning rate scheduling strategies, with focus
//! on warmup and cosine annealing commonly used in transformer training.

use crate::error::Result;

/// Learning rate scheduling strategy.
///
/// Determines how the learning rate changes during training.
///
/// # Examples
///
/// ```
/// use metal_candle::training::LRScheduler;
///
/// // Linear warmup for 1000 steps, then constant
/// let scheduler = LRScheduler::warmup_cosine(1000, 10000, 1e-4, 1e-6);
///
/// // Get LR at step 500 (during warmup)
/// let lr = scheduler.get_lr(500);
/// assert!(lr < 1e-4); // Still warming up
///
/// // Get LR at step 5000 (cosine decay)
/// let lr = scheduler.get_lr(5000);
/// assert!(lr < 1e-4 && lr > 1e-6); // Decaying
/// ```
#[derive(Debug, Clone)]
pub enum LRScheduler {
    /// Constant learning rate throughout training.
    ///
    /// # Fields
    ///
    /// * `lr` - The constant learning rate
    Constant {
        /// The constant learning rate
        lr: f32,
    },

    /// Linear warmup from 0 to `max_lr` over `warmup_steps`.
    ///
    /// After warmup, the learning rate stays constant at `max_lr`.
    ///
    /// # Fields
    ///
    /// * `warmup_steps` - Number of steps to warm up
    /// * `max_lr` - Maximum learning rate after warmup
    Linear {
        /// Number of steps to warm up
        warmup_steps: usize,
        /// Maximum learning rate after warmup
        max_lr: f32,
    },

    /// Cosine annealing from `max_lr` to `min_lr` over `total_steps`.
    ///
    /// # Fields
    ///
    /// * `total_steps` - Total number of training steps
    /// * `max_lr` - Initial learning rate
    /// * `min_lr` - Minimum learning rate
    Cosine {
        /// Total number of training steps
        total_steps: usize,
        /// Initial learning rate
        max_lr: f32,
        /// Minimum learning rate
        min_lr: f32,
    },

    /// Linear warmup followed by cosine annealing.
    ///
    /// This is the most common schedule for transformer training:
    /// - Warmup: linearly increase from 0 to `max_lr` over `warmup_steps`
    /// - Decay: cosine annealing from `max_lr` to `min_lr` over remaining steps
    ///
    /// # Fields
    ///
    /// * `warmup_steps` - Number of steps to warm up
    /// * `total_steps` - Total number of training steps
    /// * `max_lr` - Maximum learning rate (after warmup)
    /// * `min_lr` - Minimum learning rate (at end)
    WarmupCosine {
        /// Number of steps to warm up
        warmup_steps: usize,
        /// Total number of training steps
        total_steps: usize,
        /// Maximum learning rate (after warmup)
        max_lr: f32,
        /// Minimum learning rate (at end)
        min_lr: f32,
    },
}

impl LRScheduler {
    /// Creates a constant learning rate scheduler.
    #[must_use]
    pub const fn constant(lr: f32) -> Self {
        Self::Constant { lr }
    }

    /// Creates a linear warmup scheduler.
    #[must_use]
    pub const fn linear(warmup_steps: usize, max_lr: f32) -> Self {
        Self::Linear {
            warmup_steps,
            max_lr,
        }
    }

    /// Creates a cosine annealing scheduler.
    #[must_use]
    pub const fn cosine(total_steps: usize, max_lr: f32, min_lr: f32) -> Self {
        Self::Cosine {
            total_steps,
            max_lr,
            min_lr,
        }
    }

    /// Creates a warmup + cosine scheduler (most common for transformers).
    #[must_use]
    pub const fn warmup_cosine(
        warmup_steps: usize,
        total_steps: usize,
        max_lr: f32,
        min_lr: f32,
    ) -> Self {
        Self::WarmupCosine {
            warmup_steps,
            total_steps,
            max_lr,
            min_lr,
        }
    }

    /// Gets the learning rate for a given training step.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step (0-indexed)
    ///
    /// # Returns
    ///
    /// The learning rate at this step.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Steps are reasonable sizes for f32
    pub fn get_lr(&self, step: usize) -> f32 {
        match *self {
            Self::Constant { lr } => lr,

            Self::Linear {
                warmup_steps,
                max_lr,
            } => {
                if step < warmup_steps {
                    // Linear warmup: lr = max_lr * (step / warmup_steps)
                    max_lr * (step as f32 / warmup_steps as f32)
                } else {
                    max_lr
                }
            }

            Self::Cosine {
                total_steps,
                max_lr,
                min_lr,
            } => {
                let progress = (step as f32 / total_steps as f32).min(1.0);
                // Cosine annealing: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
                min_lr + 0.5 * (max_lr - min_lr) * (1.0 + (std::f32::consts::PI * progress).cos())
            }

            Self::WarmupCosine {
                warmup_steps,
                total_steps,
                max_lr,
                min_lr,
            } => {
                if step < warmup_steps {
                    // Linear warmup
                    max_lr * (step as f32 / warmup_steps as f32)
                } else {
                    // Cosine annealing from warmup_steps to total_steps
                    let decay_steps = total_steps - warmup_steps;
                    let decay_progress =
                        ((step - warmup_steps) as f32 / decay_steps as f32).min(1.0);
                    min_lr
                        + 0.5
                            * (max_lr - min_lr)
                            * (1.0 + (std::f32::consts::PI * decay_progress).cos())
                }
            }
        }
    }

    /// Validates the scheduler configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is invalid.
    pub fn validate(&self) -> Result<()> {
        match *self {
            Self::Constant { lr } => {
                if lr <= 0.0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "learning rate must be positive".to_string(),
                    }
                    .into());
                }
            }

            Self::Linear {
                warmup_steps,
                max_lr,
            } => {
                if warmup_steps == 0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "warmup_steps must be greater than 0".to_string(),
                    }
                    .into());
                }
                if max_lr <= 0.0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "max learning rate must be positive".to_string(),
                    }
                    .into());
                }
            }

            Self::Cosine {
                total_steps,
                max_lr,
                min_lr,
            } => {
                if total_steps == 0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "total_steps must be greater than 0".to_string(),
                    }
                    .into());
                }
                if max_lr <= 0.0 || min_lr < 0.0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "learning rates must be non-negative and max_lr positive"
                            .to_string(),
                    }
                    .into());
                }
                if min_lr >= max_lr {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "min_lr must be less than max_lr".to_string(),
                    }
                    .into());
                }
            }

            Self::WarmupCosine {
                warmup_steps,
                total_steps,
                max_lr,
                min_lr,
            } => {
                if warmup_steps == 0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "warmup_steps must be greater than 0".to_string(),
                    }
                    .into());
                }
                if total_steps <= warmup_steps {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "total_steps must be greater than warmup_steps".to_string(),
                    }
                    .into());
                }
                if max_lr <= 0.0 || min_lr < 0.0 {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "learning rates must be non-negative and max_lr positive"
                            .to_string(),
                    }
                    .into());
                }
                if min_lr >= max_lr {
                    return Err(crate::error::TrainingError::InvalidConfig {
                        reason: "min_lr must be less than max_lr".to_string(),
                    }
                    .into());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let scheduler = LRScheduler::constant(1e-4);

        let lr = scheduler.get_lr(0);
        assert!((lr - 1e-4).abs() < 1e-9, "Expected 1e-4, got {lr}");
        let lr = scheduler.get_lr(100);
        assert!((lr - 1e-4).abs() < 1e-9, "Expected 1e-4, got {lr}");
        let lr = scheduler.get_lr(1000);
        assert!((lr - 1e-4).abs() < 1e-9, "Expected 1e-4, got {lr}");
    }

    #[test]
    fn test_linear_warmup() {
        let scheduler = LRScheduler::linear(1000, 1e-3);

        // At step 0, LR should be 0
        let lr = scheduler.get_lr(0);
        assert!((lr - 0.0).abs() < 1e-9, "Expected 0.0, got {lr}");

        // At step 500 (halfway), LR should be max_lr / 2
        let lr = scheduler.get_lr(500);
        assert!((lr - 5e-4).abs() < 1e-7, "lr = {lr}");

        // At step 1000, LR should be max_lr
        let lr = scheduler.get_lr(1000);
        assert!((lr - 1e-3).abs() < 1e-9, "Expected 1e-3, got {lr}");

        // After warmup, LR stays constant
        let lr = scheduler.get_lr(2000);
        assert!((lr - 1e-3).abs() < 1e-9, "Expected 1e-3, got {lr}");
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = LRScheduler::cosine(10_000, 1e-3, 1e-5);

        // At step 0, LR should be max_lr
        let lr0 = scheduler.get_lr(0);
        assert!((lr0 - 1e-3).abs() < 1e-7, "lr0 = {lr0}");

        // At step 5000 (halfway), LR should be roughly halfway
        let lr_mid = scheduler.get_lr(5000);
        assert!(lr_mid > 1e-5 && lr_mid < 1e-3, "lr_mid = {lr_mid}");

        // At step 10000 (end), LR should approach min_lr
        let lr_end = scheduler.get_lr(10_000);
        assert!((lr_end - 1e-5).abs() < 1e-4, "lr_end = {lr_end}");
    }

    #[test]
    fn test_warmup_cosine() {
        let scheduler = LRScheduler::warmup_cosine(1000, 10_000, 1e-3, 1e-5);

        // During warmup (step 0-1000)
        let lr = scheduler.get_lr(0);
        assert!((lr - 0.0).abs() < 1e-9, "Expected 0.0, got {lr}");

        let lr_500 = scheduler.get_lr(500);
        assert!((lr_500 - 5e-4).abs() < 1e-7, "lr_500 = {lr_500}");

        let lr_1000 = scheduler.get_lr(1000);
        assert!((lr_1000 - 1e-3).abs() < 1e-7, "lr_1000 = {lr_1000}");

        // After warmup, cosine decay
        let lr_5000 = scheduler.get_lr(5000);
        assert!(
            lr_5000 < 1e-3 && lr_5000 > 1e-5,
            "lr_5000 = {lr_5000} should be decaying"
        );

        // Near end
        let lr_9000 = scheduler.get_lr(9000);
        assert!(lr_9000 < lr_5000, "lr_9000 = {lr_9000}");
    }

    #[test]
    fn test_scheduler_validation() {
        // Valid schedulers
        assert!(LRScheduler::constant(1e-4).validate().is_ok());
        assert!(LRScheduler::linear(1000, 1e-3).validate().is_ok());
        assert!(LRScheduler::cosine(10_000, 1e-3, 1e-5).validate().is_ok());
        assert!(LRScheduler::warmup_cosine(1000, 10_000, 1e-3, 1e-5)
            .validate()
            .is_ok());

        // Invalid: negative learning rate
        assert!(LRScheduler::constant(-1e-4).validate().is_err());

        // Invalid: zero warmup steps
        assert!(LRScheduler::linear(0, 1e-3).validate().is_err());

        // Invalid: min_lr >= max_lr
        assert!(LRScheduler::cosine(10_000, 1e-5, 1e-3).validate().is_err());

        // Invalid: total_steps <= warmup_steps
        assert!(LRScheduler::warmup_cosine(1000, 500, 1e-3, 1e-5)
            .validate()
            .is_err());
    }

    #[test]
    fn test_warmup_cosine_monotonic_decay() {
        let scheduler = LRScheduler::warmup_cosine(100, 1000, 1e-3, 1e-5);

        // LR should increase during warmup
        let mut prev_lr = scheduler.get_lr(0);
        for step in 1..100 {
            let lr = scheduler.get_lr(step);
            assert!(lr >= prev_lr, "step={step}, lr={lr}, prev={prev_lr}");
            prev_lr = lr;
        }

        // LR should decrease after warmup
        prev_lr = scheduler.get_lr(100);
        for step in 101..1000 {
            let lr = scheduler.get_lr(step);
            assert!(lr <= prev_lr, "step={step}, lr={lr}, prev={prev_lr}");
            prev_lr = lr;
        }
    }

    #[test]
    fn test_cosine_bounds() {
        let scheduler = LRScheduler::cosine(1000, 1e-3, 1e-5);

        // Check all LRs are within bounds
        for step in 0..=1000 {
            let lr = scheduler.get_lr(step);
            assert!(
                (1e-5..=1e-3).contains(&lr),
                "step={step}, lr={lr} out of bounds"
            );
        }
    }
}
