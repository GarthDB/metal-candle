//! Integration tests for Qwen model hot-swapping with ApplyAdapter trait.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use metal_candle::models::{qwen::Qwen, ModelConfig};
use metal_candle::training::{ApplyAdapter, LoRAAdapter, LoRAAdapterConfig, TargetModule};
use metal_candle::Result;
use std::sync::Arc;

/// Helper to create a small Qwen model for testing.
fn create_test_qwen(device: &Device) -> Result<Qwen> {
    let config = ModelConfig {
        architectures: vec!["qwen2".to_string()],
        vocab_size: 1000,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: Some(2),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        torch_dtype: Some("float32".to_string()),
    };
    let vb = VarBuilder::zeros(DType::F32, device);
    Qwen::new(&config, vb)
}

/// Helper to create a LoRA adapter for testing.
/// Dimensions match the test Qwen model: hidden_size=128, intermediate_size=256, 2 layers
/// NOTE: Currently only OProj is fully supported in the simplified implementation
fn create_test_adapter(device: &Device) -> Result<LoRAAdapter> {
    let config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        // Only use OProj for now - DownProj requires hooking into MLP internals
        target_modules: vec![TargetModule::OProj],
    };
    // in_features=128 (hidden_size), out_features=256 (intermediate_size), num_layers=2
    LoRAAdapter::new(128, 256, 2, &config, device)
}

#[test]
fn test_qwen_hotswap_during_inference() -> Result<()> {
    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?;

    // Create input tokens
    let input_ids = Tensor::new(&[1u32, 2, 3, 4], &device)?.unsqueeze(0)?;

    // Run forward pass without adapter
    let baseline_output = model.forward(&input_ids, None)?;
    assert!(!model.has_adapter());

    // Create and apply first adapter
    let adapter1 = Arc::new(create_test_adapter(&device)?);
    model.apply_adapter(Arc::clone(&adapter1))?;
    assert!(model.has_adapter());

    // Run forward pass with adapter
    let adapter1_output = model.forward(&input_ids, None)?;

    // Outputs should have same shape but potentially different values
    assert_eq!(baseline_output.dims(), adapter1_output.dims());

    // Create and apply second adapter (hot-swap)
    let adapter2 = Arc::new(create_test_adapter(&device)?);
    model.apply_adapter(Arc::clone(&adapter2))?;
    assert!(model.has_adapter());

    // Run forward pass with second adapter
    let adapter2_output = model.forward(&input_ids, None)?;
    assert_eq!(baseline_output.dims(), adapter2_output.dims());

    // Remove adapter
    model.remove_adapter()?;
    assert!(!model.has_adapter());

    // Run forward pass after removal (should match baseline behavior)
    let post_removal_output = model.forward(&input_ids, None)?;
    assert_eq!(baseline_output.dims(), post_removal_output.dims());

    Ok(())
}

#[test]
fn test_qwen_adapter_persistence_across_batches() -> Result<()> {
    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?;

    // Create and apply adapter
    let adapter = Arc::new(create_test_adapter(&device)?);
    model.apply_adapter(Arc::clone(&adapter))?;

    // Run multiple batches with same adapter
    for batch_id in 0..5 {
        let input_ids = Tensor::new(&[(batch_id + 1) as u32, 2, 3], &device)?.unsqueeze(0)?;
        let output = model.forward(&input_ids, None)?;

        // Verify adapter is still active
        assert!(model.has_adapter());

        // Verify output shape is correct
        assert_eq!(output.dims(), &[1, 3, 1000]); // [batch, seq_len, vocab_size]
    }

    Ok(())
}

#[test]
fn test_qwen_multiple_rapid_adapter_swaps() -> Result<()> {
    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?;
    let input_ids = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;

    // Create multiple adapters
    let adapters: Vec<Arc<LoRAAdapter>> = (0..10)
        .map(|_| Arc::new(create_test_adapter(&device).unwrap()))
        .collect();

    // Rapidly swap adapters
    for (i, adapter) in adapters.iter().enumerate() {
        model.apply_adapter(Arc::clone(adapter))?;
        assert!(model.has_adapter());

        // Run forward pass
        let output = model.forward(&input_ids, None)?;
        assert_eq!(output.dims(), &[1, 3, 1000]);

        // Occasionally remove adapter
        if i % 3 == 0 {
            model.remove_adapter()?;
            assert!(!model.has_adapter());

            // Verify forward pass still works without adapter
            let output_no_adapter = model.forward(&input_ids, None)?;
            assert_eq!(output_no_adapter.dims(), &[1, 3, 1000]);
        }
    }

    Ok(())
}

#[test]
fn test_qwen_adapter_with_registry_integration() -> Result<()> {
    use metal_candle::training::AdapterRegistry;

    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?;
    let mut registry = AdapterRegistry::new();

    // Create and register multiple adapters
    let adapter1 = create_test_adapter(&device)?;
    let adapter2 = create_test_adapter(&device)?;

    registry.add_adapter("code-assistant".to_string(), adapter1)?;
    registry.add_adapter("chat".to_string(), adapter2)?;

    // Activate first adapter in registry
    registry.activate("code-assistant")?;

    // Apply to model
    if let Some(active) = registry.get_active() {
        model.apply_adapter(Arc::clone(active))?;
    }
    assert!(model.has_adapter());

    // Run inference with first adapter
    let input_ids = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;
    let output1 = model.forward(&input_ids, None)?;

    // Switch adapter in registry
    registry.activate("chat")?;

    // Apply new adapter to model (hot-swap)
    if let Some(active) = registry.get_active() {
        model.apply_adapter(Arc::clone(active))?;
    }
    assert!(model.has_adapter());

    // Run inference with second adapter
    let output2 = model.forward(&input_ids, None)?;

    // Verify both outputs have correct shape
    assert_eq!(output1.dims(), output2.dims());
    assert_eq!(output1.dims(), &[1, 3, 1000]);

    Ok(())
}

#[test]
fn test_qwen_adapter_layer_mismatch_error() -> Result<()> {
    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?; // 2 layers

    // Create adapter with wrong number of layers (4 instead of 2)
    let config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::OProj],
    };
    let bad_adapter = Arc::new(LoRAAdapter::new(128, 256, 4, &config, &device)?);

    // Attempt to apply adapter with mismatched layers
    let result = model.apply_adapter(bad_adapter);

    // Should return an error
    assert!(result.is_err());
    assert!(!model.has_adapter());

    Ok(())
}

#[test]
fn test_qwen_adapter_with_different_target_modules() -> Result<()> {
    let device = Device::Cpu;
    let mut model = create_test_qwen(&device)?;
    let input_ids = Tensor::new(&[1u32, 2, 3], &device)?.unsqueeze(0)?;

    // Test with OProj (currently supported)
    let config_oproj = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::OProj],
    };
    let adapter_oproj = Arc::new(LoRAAdapter::new(128, 256, 2, &config_oproj, &device)?);
    model.apply_adapter(Arc::clone(&adapter_oproj))?;
    let output_oproj = model.forward(&input_ids, None)?;
    assert_eq!(output_oproj.dims(), &[1, 3, 1000]);

    // Test removing and re-applying
    model.remove_adapter()?;
    assert!(!model.has_adapter());

    model.apply_adapter(Arc::clone(&adapter_oproj))?;
    let output_reapplied = model.forward(&input_ids, None)?;
    assert_eq!(output_reapplied.dims(), &[1, 3, 1000]);

    Ok(())
}
