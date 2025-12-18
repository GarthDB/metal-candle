//! Tests for `ApplyAdapter` trait implementation on Qwen model.

#[cfg(test)]
mod tests {
    use crate::models::{ModelConfig, Qwen};
    use crate::training::{ApplyAdapter, LoRAAdapter, LoRAAdapterConfig, TargetModule};
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::sync::Arc;

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            architectures: vec!["qwen2".to_string()],
            vocab_size: 1000,
            hidden_size: 128,
            intermediate_size: 512,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            torch_dtype: Some("float32".to_string()),
        }
    }

    #[test]
    fn test_apply_adapter_trait() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut model = Qwen::new(&config, vb).unwrap();

        // Initially no adapter
        assert!(!model.has_adapter());

        // Create and apply adapter
        let adapter_config = LoRAAdapterConfig::default();
        let adapter = LoRAAdapter::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            &adapter_config,
            &device,
        )
        .unwrap();

        let adapter_arc = Arc::new(adapter);
        assert!(model.apply_adapter(adapter_arc.clone()).is_ok());

        // Now has adapter
        assert!(model.has_adapter());

        // Remove adapter
        assert!(model.remove_adapter().is_ok());

        // No longer has adapter
        assert!(!model.has_adapter());
    }

    #[test]
    fn test_apply_adapter_replace() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut model = Qwen::new(&config, vb).unwrap();

        // Apply first adapter
        let adapter_config1 = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::QProj],
        };
        let adapter1 = Arc::new(
            LoRAAdapter::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                &adapter_config1,
                &device,
            )
            .unwrap(),
        );

        model.apply_adapter(adapter1).unwrap();
        assert!(model.has_adapter());

        // Apply second adapter (should replace first)
        let adapter_config2 = LoRAAdapterConfig {
            rank: 8,
            alpha: 16.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::VProj],
        };
        let adapter2 = Arc::new(
            LoRAAdapter::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                &adapter_config2,
                &device,
            )
            .unwrap(),
        );

        model.apply_adapter(adapter2).unwrap();
        assert!(model.has_adapter());
    }

    #[test]
    fn test_forward_with_adapter() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut model = Qwen::new(&config, vb).unwrap();
        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        // Forward without adapter
        let logits_no_adapter = model.forward(&input_ids, None).unwrap();

        // Apply adapter (only to OProj which takes hidden_size input)
        let adapter_config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::OProj],
        };
        let adapter = Arc::new(
            LoRAAdapter::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                &adapter_config,
                &device,
            )
            .unwrap(),
        );

        model.apply_adapter(adapter).unwrap();

        // Forward with adapter (should not fail)
        let logits_with_adapter = model.forward(&input_ids, None).unwrap();

        // Verify output shapes match
        assert_eq!(logits_no_adapter.dims(), logits_with_adapter.dims());

        // Verify dimensions
        let (batch, seq, vocab) = logits_with_adapter.dims3().unwrap();
        assert_eq!(batch, 1);
        assert_eq!(seq, 4);
        assert_eq!(vocab, config.vocab_size);
    }

    #[test]
    fn test_remove_adapter_restores_behavior() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut model = Qwen::new(&config, vb).unwrap();
        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        // Forward without adapter
        let logits_before = model.forward(&input_ids, None).unwrap();

        // Apply and remove adapter
        let adapter_config = LoRAAdapterConfig {
            rank: 4,
            alpha: 8.0,
            dropout: 0.0,
            target_modules: vec![TargetModule::OProj],
        };
        let adapter = Arc::new(
            LoRAAdapter::new(
                config.hidden_size,
                config.intermediate_size,
                config.num_hidden_layers,
                &adapter_config,
                &device,
            )
            .unwrap(),
        );

        model.apply_adapter(adapter).unwrap();
        model.remove_adapter().unwrap();

        // Forward after removal
        let logits_after = model.forward(&input_ids, None).unwrap();

        // Shapes should match
        assert_eq!(logits_before.dims(), logits_after.dims());
    }

    #[test]
    fn test_multiple_adapter_swaps() {
        let config = create_test_config();
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);

        let mut model = Qwen::new(&config, vb).unwrap();
        let input_ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();

        // Create multiple adapters
        let adapter_configs = vec![
            LoRAAdapterConfig {
                rank: 4,
                alpha: 8.0,
                dropout: 0.0,
                target_modules: vec![TargetModule::OProj],
            },
            LoRAAdapterConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
                target_modules: vec![TargetModule::OProj],
            },
            LoRAAdapterConfig {
                rank: 16,
                alpha: 32.0,
                dropout: 0.0,
                target_modules: vec![TargetModule::OProj],
            },
        ];

        // Swap adapters multiple times
        for adapter_config in &adapter_configs {
            let adapter = Arc::new(
                LoRAAdapter::new(
                    config.hidden_size,
                    config.intermediate_size,
                    config.num_hidden_layers,
                    adapter_config,
                    &device,
                )
                .unwrap(),
            );

            model.apply_adapter(adapter).unwrap();
            assert!(model.has_adapter());

            // Forward pass should work
            let logits = model.forward(&input_ids, None);
            assert!(logits.is_ok());
        }

        // Remove final adapter
        model.remove_adapter().unwrap();
        assert!(!model.has_adapter());
    }
}
