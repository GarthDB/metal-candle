//! Trait for applying `LoRA` adapters to models.
//!
//! This module defines the interface for hot-swapping `LoRA` adapters
//! on models without reloading the base weights.

use super::adapter::LoRAAdapter;
use crate::error::Result;

/// Trait for models that support `LoRA` adapter hot-swapping.
///
/// Implementing this trait allows a model to dynamically apply and remove
/// `LoRA` adapters without reloading the base model weights.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::{ApplyAdapter, LoRAAdapter};
///
/// # fn example<M: ApplyAdapter>(mut model: M, adapter: &LoRAAdapter) -> Result<(), Box<dyn std::error::Error>> {
/// // Apply adapter
/// model.apply_adapter(adapter)?;
///
/// // Use model with adapter...
///
/// // Remove adapter
/// model.remove_adapter()?;
/// # Ok(())
/// # }
/// ```
pub trait ApplyAdapter {
    /// Applies a `LoRA` adapter to the model.
    ///
    /// The adapter's weights are integrated into the model's forward pass.
    /// Any previously applied adapter is replaced.
    ///
    /// # Arguments
    ///
    /// * `adapter` - The `LoRA` adapter to apply
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The adapter structure doesn't match the model
    /// - Adapter application fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::{ApplyAdapter, LoRAAdapter};
    /// # fn example<M: ApplyAdapter>(mut model: M, adapter: &LoRAAdapter) -> Result<(), Box<dyn std::error::Error>> {
    /// model.apply_adapter(adapter)?;
    /// # Ok(())
    /// # }
    /// ```
    fn apply_adapter(&mut self, adapter: &LoRAAdapter) -> Result<()>;

    /// Removes the currently applied `LoRA` adapter.
    ///
    /// After calling this, the model behaves as if no adapter is applied.
    ///
    /// # Errors
    ///
    /// Returns an error if adapter removal fails.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::ApplyAdapter;
    /// # fn example<M: ApplyAdapter>(mut model: M) -> Result<(), Box<dyn std::error::Error>> {
    /// model.remove_adapter()?;
    /// # Ok(())
    /// # }
    /// ```
    fn remove_adapter(&mut self) -> Result<()>;

    /// Returns `true` if an adapter is currently applied.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::ApplyAdapter;
    /// # fn example<M: ApplyAdapter>(model: &M) {
    /// if model.has_adapter() {
    ///     println!("Model has an active adapter");
    /// }
    /// # }
    /// ```
    fn has_adapter(&self) -> bool;
}

// Note on Implementation
//
// Full implementation of `ApplyAdapter` for Qwen and other models requires:
//
// 1. **Model Refactoring**: Store base weights and adapter weights separately
// 2. **Forward Pass Integration**: Conditionally apply adapter in forward pass
// 3. **Memory Management**: Efficient adapter weight storage and transfer
//
// This is planned for v1.3.1 as a follow-up enhancement. The current v1.3.0
// release provides the `AdapterRegistry` for managing multiple adapters,
// which can be used with manual model reloading as an interim solution.
