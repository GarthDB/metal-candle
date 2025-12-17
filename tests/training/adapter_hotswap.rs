//! Integration tests for adapter hot-swapping workflow.

use candle_core::Device;
use metal_candle::training::{AdapterRegistry, LoRAAdapter, LoRAAdapterConfig, TargetModule};
use metal_candle::Result;
use tempfile::NamedTempFile;

#[test]
fn test_adapter_registry_workflow() -> Result<()> {
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };

    // Create registry
    let mut registry = AdapterRegistry::new();
    assert!(registry.is_empty());

    // Create and add first adapter
    let adapter1 = LoRAAdapter::new(256, 1024, 6, &config, &device)?;
    registry.add_adapter("code-assistant".to_string(), adapter1)?;

    // Create and add second adapter
    let adapter2 = LoRAAdapter::new(256, 1024, 6, &config, &device)?;
    registry.add_adapter("chat".to_string(), adapter2)?;

    // Verify both are loaded
    assert_eq!(registry.len(), 2);
    assert!(registry.list_adapters().contains(&"code-assistant".to_string()));
    assert!(registry.list_adapters().contains(&"chat".to_string()));

    // Activate first adapter
    registry.activate("code-assistant")?;
    assert_eq!(registry.active_adapter(), Some("code-assistant"));
    assert!(registry.get_active().is_some());

    // Switch to second adapter
    registry.activate("chat")?;
    assert_eq!(registry.active_adapter(), Some("chat"));

    // Deactivate
    registry.deactivate();
    assert!(registry.active_adapter().is_none());
    assert!(registry.get_active().is_none());

    // Unload one adapter
    registry.unload_adapter("code-assistant")?;
    assert_eq!(registry.len(), 1);

    Ok(())
}

#[test]
fn test_adapter_save_and_load_workflow() -> Result<()> {
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj],
    };

    // Create adapter
    let adapter = LoRAAdapter::new(128, 512, 4, &config, &device)?;

    // Save to temporary file
    let temp_file = NamedTempFile::new().unwrap();
    let checkpoint_path = temp_file.path();
    
    metal_candle::training::checkpoint::save_checkpoint(&adapter, checkpoint_path, None)?;

    // Create new adapter with same config
    let mut loaded_adapter = LoRAAdapter::new(128, 512, 4, &config, &device)?;

    // Load weights
    metal_candle::training::checkpoint::load_checkpoint(&mut loaded_adapter, checkpoint_path)?;

    // Verify adapter loaded successfully
    assert_eq!(
        adapter.num_trainable_parameters(),
        loaded_adapter.num_trainable_parameters()
    );

    Ok(())
}

#[test]
fn test_adapter_registry_with_checkpoint() -> Result<()> {
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj],
    };

    // Create and save adapter
    let adapter = LoRAAdapter::new(64, 256, 2, &config, &device)?;
    let temp_file = NamedTempFile::new().unwrap();
    let checkpoint_path = temp_file.path();
    metal_candle::training::checkpoint::save_checkpoint(&adapter, checkpoint_path, None)?;

    // Create registry and load from checkpoint
    let mut registry = AdapterRegistry::new();
    let loaded_adapter = LoRAAdapter::new(64, 256, 2, &config, &device)?;
    registry.load_adapter_from_checkpoint(
        "saved-adapter".to_string(),
        loaded_adapter,
        checkpoint_path,
    )?;

    // Verify adapter is in registry
    assert_eq!(registry.len(), 1);
    assert!(registry.get_adapter("saved-adapter").is_some());

    // Activate and use
    registry.activate("saved-adapter")?;
    assert_eq!(registry.active_adapter(), Some("saved-adapter"));

    Ok(())
}

#[test]
fn test_multiple_adapter_switching() -> Result<()> {
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj],
    };

    let mut registry = AdapterRegistry::new();

    // Add multiple adapters
    for i in 0..5 {
        let adapter = LoRAAdapter::new(32, 128, 2, &config, &device)?;
        registry.add_adapter(format!("adapter-{}", i), adapter)?;
    }

    assert_eq!(registry.len(), 5);

    // Rapidly switch between adapters
    for i in 0..5 {
        let name = format!("adapter-{}", i);
        registry.activate(&name)?;
        assert_eq!(registry.active_adapter(), Some(name.as_str()));
    }

    // Deactivate and verify
    registry.deactivate();
    assert!(registry.active_adapter().is_none());

    Ok(())
}

#[test]
fn test_adapter_error_handling() -> Result<()> {
    let mut registry = AdapterRegistry::new();

    // Try to activate non-existent adapter
    let result = registry.activate("nonexistent");
    assert!(result.is_err());

    // Try to unload non-existent adapter
    let result = registry.unload_adapter("nonexistent");
    assert!(result.is_err());

    // Try to add duplicate adapter
    let device = Device::Cpu;
    let config = LoRAAdapterConfig::default();
    let adapter1 = LoRAAdapter::new(32, 128, 2, &config, &device)?;
    let adapter2 = LoRAAdapter::new(32, 128, 2, &config, &device)?;

    registry.add_adapter("test".to_string(), adapter1)?;
    let result = registry.add_adapter("test".to_string(), adapter2);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_adapter_memory_efficiency() -> Result<()> {
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };

    // Create adapter
    let adapter = LoRAAdapter::new(512, 2048, 12, &config, &device)?;
    let params = adapter.num_trainable_parameters();

    // Verify parameter count is reasonable for LoRA
    // For rank=8, 2 target modules, 12 layers:
    // Each module: (512 * 8 + 8 * 512) = 8,192 params per module
    // Total: 8,192 * 2 modules * 12 layers = 196,608 params
    assert!(params > 150_000 && params < 250_000, "Expected ~196k params, got {}", params);

    // Add to registry (no duplication of base model)
    let mut registry = AdapterRegistry::new();
    registry.add_adapter("test".to_string(), adapter)?;

    // Verify adapter is accessible
    assert!(registry.get_adapter("test").is_some());

    Ok(())
}

