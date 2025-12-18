//! Registry for managing multiple `LoRA` adapters.
//!
//! This module provides functionality to load, manage, and hot-swap multiple
//! `LoRA` adapters without reloading the base model.

use super::adapter::LoRAAdapter;
use crate::error::{Result, TrainingError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Registry for managing multiple `LoRA` adapters.
///
/// Allows loading, unloading, and activating different `LoRA` adapters
/// without reloading the base model. This enables efficient adapter
/// hot-swapping for production deployments.
///
/// # Important: Current Limitations (v1.3.0)
///
/// **Note**: In v1.3.0, the registry manages adapter storage and activation state,
/// but does not automatically apply adapters to models during inference. Full
/// hot-swapping with automatic model integration requires the [`ApplyAdapter`](super::ApplyAdapter)
/// trait to be implemented for your model, which is planned for v1.3.1.
///
/// **Current workflow** (v1.3.0):
/// - Use the registry to organize and switch between adapters
/// - Manually integrate the active adapter with your model's forward pass
/// - Adapter switching is instant (<100ms) but requires manual wiring
///
/// **Future workflow** (v1.3.1+):
/// - Models implementing [`ApplyAdapter`](super::ApplyAdapter) will automatically use the active adapter
/// - Call `model.apply_adapter(registry.get_active()?)` for seamless integration
/// - True zero-downtime hot-swapping without manual model updates
///
/// # Examples
///
/// ```no_run
/// use metal_candle::training::AdapterRegistry;
/// use std::sync::Arc;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create registry (base_model would be your actual model)
/// // let registry = AdapterRegistry::new();
///
/// // Load adapters
/// // registry.load_adapter("code-assistant".to_string(), "adapters/code.safetensors")?;
/// // registry.load_adapter("chat".to_string(), "adapters/chat.safetensors")?;
///
/// // Activate an adapter
/// // registry.activate("code-assistant")?;
///
/// // In v1.3.0: Manually integrate with your model
/// // let active_adapter = registry.get_active().expect("No active adapter");
/// // // Use active_adapter in your model's forward pass
///
/// // In v1.3.1+: Automatic integration (once ApplyAdapter is implemented)
/// // model.apply_adapter(registry.get_active()?)?;
///
/// // List available adapters
/// // let adapters = registry.list_adapters();
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AdapterRegistry {
    /// Loaded adapters indexed by name
    adapters: HashMap<String, Arc<LoRAAdapter>>,

    /// Currently active adapter name
    active: Option<String>,
}

impl AdapterRegistry {
    /// Creates a new empty adapter registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert!(registry.list_adapters().is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            adapters: HashMap::new(),
            active: None,
        }
    }

    /// Loads a `LoRA` adapter from a checkpoint file.
    ///
    /// The adapter is loaded into memory but not activated. Use [`activate`](Self::activate)
    /// to make it the active adapter.
    ///
    /// **Note**: This method requires an adapter structure to be provided. In practice,
    /// you would create an adapter with the appropriate configuration and then load
    /// the weights from the checkpoint.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this adapter
    /// * `adapter` - Pre-configured adapter to load weights into
    /// * `path` - Path to the adapter checkpoint file (safetensors format)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - An adapter with this name already exists
    /// - The checkpoint file cannot be loaded
    /// - The checkpoint format is invalid
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::{AdapterRegistry, LoRAAdapter, LoRAAdapterConfig};
    /// # use candle_core::Device;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let config = LoRAAdapterConfig::default();
    /// let mut adapter = LoRAAdapter::new(768, 3072, 12, &config, &device)?;
    ///
    /// let mut registry = AdapterRegistry::new();
    /// registry.load_adapter_from_checkpoint("my-adapter".to_string(), adapter, "path/to/adapter.safetensors")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn load_adapter_from_checkpoint(
        &mut self,
        name: String,
        mut adapter: LoRAAdapter,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        // Check if adapter already exists
        if self.adapters.contains_key(&name) {
            return Err(TrainingError::InvalidConfig {
                reason: format!("Adapter '{name}' already exists"),
            }
            .into());
        }

        // Load weights from checkpoint
        super::checkpoint::load_checkpoint(&mut adapter, path)?;

        // Add to registry
        self.adapters.insert(name, Arc::new(adapter));
        Ok(())
    }

    /// Adds a pre-configured adapter to the registry.
    ///
    /// This is useful when you've already created and configured an adapter
    /// and want to add it to the registry without loading from a file.
    ///
    /// # Arguments
    ///
    /// * `name` - Unique identifier for this adapter
    /// * `adapter` - The adapter to add
    ///
    /// # Errors
    ///
    /// Returns an error if an adapter with this name already exists.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::{AdapterRegistry, LoRAAdapter, LoRAAdapterConfig};
    /// # use candle_core::Device;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let device = Device::Cpu;
    /// let config = LoRAAdapterConfig::default();
    /// let adapter = LoRAAdapter::new(768, 3072, 12, &config, &device)?;
    ///
    /// let mut registry = AdapterRegistry::new();
    /// registry.add_adapter("my-adapter".to_string(), adapter)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn add_adapter(&mut self, name: String, adapter: LoRAAdapter) -> Result<()> {
        // Check if adapter already exists
        if self.adapters.contains_key(&name) {
            return Err(TrainingError::InvalidConfig {
                reason: format!("Adapter '{name}' already exists"),
            }
            .into());
        }

        self.adapters.insert(name, Arc::new(adapter));
        Ok(())
    }

    /// Unloads an adapter from the registry.
    ///
    /// If the adapter is currently active, it is deactivated first.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the adapter to unload
    ///
    /// # Errors
    ///
    /// Returns an error if the adapter does not exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::AdapterRegistry;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut registry = AdapterRegistry::new();
    /// // registry.load_adapter("my-adapter".to_string(), "path/to/adapter.safetensors")?;
    /// // registry.unload_adapter("my-adapter")?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn unload_adapter(&mut self, name: &str) -> Result<()> {
        if !self.adapters.contains_key(name) {
            return Err(TrainingError::InvalidConfig {
                reason: format!("Adapter '{name}' not found"),
            }
            .into());
        }

        // Deactivate if this is the active adapter
        if self.active.as_deref() == Some(name) {
            self.active = None;
        }

        self.adapters.remove(name);
        Ok(())
    }

    /// Activates an adapter.
    ///
    /// The specified adapter becomes the active adapter. Any previously
    /// active adapter is deactivated.
    ///
    /// **Note (v1.3.0)**: This method updates the registry's internal state but does
    /// not automatically apply the adapter to your model. You must manually integrate
    /// the active adapter (retrieved via [`get_active()`](Self::get_active)) with your
    /// model's forward pass. Full automatic integration requires [`ApplyAdapter`](super::ApplyAdapter)
    /// trait implementation, planned for v1.3.1.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the adapter to activate
    ///
    /// # Errors
    ///
    /// Returns an error if the adapter does not exist.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::training::AdapterRegistry;
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut registry = AdapterRegistry::new();
    /// // registry.load_adapter("my-adapter".to_string(), "path/to/adapter.safetensors")?;
    /// // registry.activate("my-adapter")?;
    ///
    /// // v1.3.0: Manual integration
    /// // let active = registry.get_active().expect("No active adapter");
    /// // Use active adapter in your model...
    ///
    /// // v1.3.1+: Automatic integration
    /// // model.apply_adapter(registry.get_active()?)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn activate(&mut self, name: &str) -> Result<()> {
        if !self.adapters.contains_key(name) {
            return Err(TrainingError::InvalidConfig {
                reason: format!("Adapter '{name}' not found"),
            }
            .into());
        }

        self.active = Some(name.to_string());
        Ok(())
    }

    /// Deactivates the currently active adapter.
    ///
    /// After calling this, no adapter is active and the model behaves
    /// as if no `LoRA` adaptation is applied.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let mut registry = AdapterRegistry::new();
    /// registry.deactivate();
    /// assert!(registry.active_adapter().is_none());
    /// ```
    pub fn deactivate(&mut self) {
        self.active = None;
    }

    /// Returns the name of the currently active adapter.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert!(registry.active_adapter().is_none());
    /// ```
    #[must_use]
    pub fn active_adapter(&self) -> Option<&str> {
        self.active.as_deref()
    }

    /// Returns a reference to the currently active adapter.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert!(registry.get_active().is_none());
    /// ```
    #[must_use]
    pub fn get_active(&self) -> Option<&Arc<LoRAAdapter>> {
        self.active
            .as_ref()
            .and_then(|name| self.adapters.get(name))
    }

    /// Returns a reference to a specific adapter by name.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the adapter to retrieve
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert!(registry.get_adapter("nonexistent").is_none());
    /// ```
    #[must_use]
    pub fn get_adapter(&self, name: &str) -> Option<&Arc<LoRAAdapter>> {
        self.adapters.get(name)
    }

    /// Lists all loaded adapter names.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// let adapters = registry.list_adapters();
    /// assert!(adapters.is_empty());
    /// ```
    #[must_use]
    pub fn list_adapters(&self) -> Vec<String> {
        self.adapters.keys().cloned().collect()
    }

    /// Returns the number of loaded adapters.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert_eq!(registry.len(), 0);
    /// ```
    #[must_use]
    pub fn len(&self) -> usize {
        self.adapters.len()
    }

    /// Returns `true` if no adapters are loaded.
    ///
    /// # Examples
    ///
    /// ```
    /// use metal_candle::training::AdapterRegistry;
    ///
    /// let registry = AdapterRegistry::new();
    /// assert!(registry.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adapters.is_empty()
    }
}

impl Default for AdapterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = AdapterRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.active_adapter().is_none());
    }

    #[test]
    fn test_registry_default() {
        let registry = AdapterRegistry::default();
        assert!(registry.is_empty());
    }

    #[test]
    fn test_list_adapters_empty() {
        let registry = AdapterRegistry::new();
        assert!(registry.list_adapters().is_empty());
    }

    #[test]
    fn test_deactivate_when_none_active() {
        let mut registry = AdapterRegistry::new();
        registry.deactivate();
        assert!(registry.active_adapter().is_none());
    }

    #[test]
    fn test_get_adapter_nonexistent() {
        let registry = AdapterRegistry::new();
        assert!(registry.get_adapter("nonexistent").is_none());
    }

    #[test]
    fn test_get_active_when_none() {
        let registry = AdapterRegistry::new();
        assert!(registry.get_active().is_none());
    }

    #[test]
    fn test_unload_nonexistent_adapter() {
        let mut registry = AdapterRegistry::new();
        let result = registry.unload_adapter("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_activate_nonexistent_adapter() {
        let mut registry = AdapterRegistry::new();
        let result = registry.activate("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_add_adapter() {
        use super::super::{LoRAAdapterConfig, TargetModule};
        use candle_core::Device;

        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        let adapter = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();
        let mut registry = AdapterRegistry::new();

        // Add adapter
        let result = registry.add_adapter("test-adapter".to_string(), adapter);
        assert!(result.is_ok());

        // Verify it's in the registry
        assert_eq!(registry.len(), 1);
        assert!(registry.get_adapter("test-adapter").is_some());
    }

    #[test]
    fn test_add_duplicate_adapter() {
        use super::super::{LoRAAdapterConfig, TargetModule};
        use candle_core::Device;

        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        let adapter1 = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();
        let adapter2 = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();

        let mut registry = AdapterRegistry::new();
        registry.add_adapter("test".to_string(), adapter1).unwrap();

        // Try to add another with same name
        let result = registry.add_adapter("test".to_string(), adapter2);
        assert!(result.is_err());
    }

    #[test]
    fn test_activate_and_get_active() {
        use super::super::{LoRAAdapterConfig, TargetModule};
        use candle_core::Device;

        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        let adapter = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();
        let mut registry = AdapterRegistry::new();
        registry.add_adapter("test".to_string(), adapter).unwrap();

        // Activate
        registry.activate("test").unwrap();
        assert_eq!(registry.active_adapter(), Some("test"));

        // Get active adapter
        assert!(registry.get_active().is_some());
    }

    #[test]
    fn test_unload_active_adapter() {
        use super::super::{LoRAAdapterConfig, TargetModule};
        use candle_core::Device;

        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        let adapter = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();
        let mut registry = AdapterRegistry::new();
        registry.add_adapter("test".to_string(), adapter).unwrap();
        registry.activate("test").unwrap();

        // Unload active adapter
        registry.unload_adapter("test").unwrap();

        // Should be deactivated
        assert!(registry.active_adapter().is_none());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_list_multiple_adapters() {
        use super::super::{LoRAAdapterConfig, TargetModule};
        use candle_core::Device;

        let device = Device::Cpu;
        let config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };

        let mut registry = AdapterRegistry::new();

        // Add multiple adapters
        for i in 0..3 {
            let adapter = LoRAAdapter::new(32, 128, 2, &config, &device).unwrap();
            registry
                .add_adapter(format!("adapter-{i}"), adapter)
                .unwrap();
        }

        let adapters = registry.list_adapters();
        assert_eq!(adapters.len(), 3);
        assert_eq!(registry.len(), 3);
    }
}
