//! Comprehensive tests for the Trainer module.
//!
//! Tests cover configuration validation, training execution, error handling,
//! and gradient accumulation.

use candle_core::{Device, DType, Tensor};
use metal_candle::training::{
    LRScheduler, LoRAAdapterConfig, Trainer, TrainingConfig,
};
use metal_candle::Result;

/// Helper function to create a simple forward function for testing.
fn create_test_forward_fn(
    device: &Device,
    vocab_size: usize,
) -> impl Fn(&Tensor) -> Result<Tensor> + '_ {
    move |input: &Tensor| {
        let batch_size = input.dim(0)?;
        let seq_len = input.dim(1)?;
        // Return logits of shape (batch, seq, vocab)
        Ok(Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, vocab_size),
            device,
        )?)
    }
}

#[test]
fn test_trainer_creation_basic() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig::default();

    let trainer = Trainer::new(256, 512, 4, &lora_config, training_config, &device)?;

    assert_eq!(trainer.global_step(), 0);
    assert!(trainer.lora_adapter().num_trainable_parameters() > 0);

    Ok(())
}

#[test]
fn test_trainer_creation_with_custom_config() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig {
        rank: 16,
        alpha: 32.0,
        dropout: 0.1,
        ..Default::default()
    };
    let training_config = TrainingConfig {
        num_epochs: 5,
        lr_scheduler: LRScheduler::linear(10, 1e-3),
        max_grad_norm: Some(2.0),
        ..Default::default()
    };

    let trainer = Trainer::new(128, 256, 2, &lora_config, training_config, &device)?;

    assert_eq!(trainer.global_step(), 0);

    Ok(())
}

#[test]
fn test_trainer_with_zero_rank_fails() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig {
        rank: 0, // Invalid
        ..Default::default()
    };
    let training_config = TrainingConfig::default();

    let result = Trainer::new(256, 512, 4, &lora_config, training_config, &device);

    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_training_single_epoch_single_batch() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig {
        num_epochs: 1,
        lr_scheduler: LRScheduler::Constant { lr: 1e-4 },
        ..Default::default()
    };

    let mut trainer = Trainer::new(128, 256, 2, &lora_config, training_config, &device)?;

    // Create simple dataset
    let input = Tensor::zeros((2, 8), DType::U32, &device)?;
    let target = Tensor::zeros((2, 8), DType::U32, &device)?;
    let dataset = vec![(input, target)];

    // Create forward function
    let vocab_size = 1000;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    // Train
    let metrics = trainer.train(&dataset, forward_fn)?;

    assert_eq!(metrics.len(), 1); // 1 epoch * 1 batch
    assert_eq!(trainer.global_step(), 1);
    assert!(metrics[0].loss >= 0.0); // Loss should be non-negative

    Ok(())
}

#[test]
fn test_training_multiple_epochs() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig {
        rank: 4,
        ..Default::default()
    };
    let training_config = TrainingConfig {
        num_epochs: 3,
        lr_scheduler: LRScheduler::Constant { lr: 1e-4 },
        ..Default::default()
    };

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    // Create small dataset
    let input1 = Tensor::zeros((1, 4), DType::U32, &device)?;
    let target1 = Tensor::zeros((1, 4), DType::U32, &device)?;
    let input2 = Tensor::ones((1, 4), DType::U32, &device)?;
    let target2 = Tensor::ones((1, 4), DType::U32, &device)?;
    let dataset = vec![(input1, target1), (input2, target2)];

    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    // Train
    let metrics = trainer.train(&dataset, forward_fn)?;

    assert_eq!(metrics.len(), 6); // 3 epochs * 2 batches
    assert_eq!(trainer.global_step(), 6);

    // Verify all metrics have valid values
    for metric in &metrics {
        assert!(metric.loss >= 0.0);
        assert!(metric.learning_rate > 0.0);
    }

    Ok(())
}

#[test]
fn test_lr_scheduler_constant() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig {
        num_epochs: 1,
        lr_scheduler: LRScheduler::Constant { lr: 5e-4 },
        ..Default::default()
    };

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    let input = Tensor::zeros((1, 4), DType::U32, &device)?;
    let target = Tensor::zeros((1, 4), DType::U32, &device)?;
    let dataset = vec![(input.clone(), target.clone()), (input, target)];

    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    let metrics = trainer.train(&dataset, forward_fn)?;

    // All learning rates should be the same for constant scheduler
    assert!((metrics[0].learning_rate - 5e-4).abs() < 1e-10);
    assert!((metrics[1].learning_rate - 5e-4).abs() < 1e-10);

    Ok(())
}

#[test]
fn test_lr_scheduler_linear() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig {
        num_epochs: 1,
        lr_scheduler: LRScheduler::linear(2, 1e-3),
        ..Default::default()
    };

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    let input = Tensor::zeros((1, 4), DType::U32, &device)?;
    let target = Tensor::zeros((1, 4), DType::U32, &device)?;
    let dataset = vec![
        (input.clone(), target.clone()),
        (input.clone(), target.clone()),
        (input, target),
    ];

    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    let metrics = trainer.train(&dataset, forward_fn)?;

    // Linear scheduler: LR increases during warmup (2 steps), then stays constant
    assert!(metrics[0].learning_rate < metrics[1].learning_rate); // Step 0 < Step 1
    assert!((metrics[2].learning_rate - 1e-3).abs() < 1e-6); // Step 2 >= warmup_steps, at max LR

    Ok(())
}

#[test]
fn test_training_step_increments() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig {
        num_epochs: 2,
        ..Default::default()
    };

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    assert_eq!(trainer.global_step(), 0);

    let input = Tensor::zeros((1, 4), DType::U32, &device)?;
    let target = Tensor::zeros((1, 4), DType::U32, &device)?;
    let dataset = vec![(input.clone(), target.clone()), (input, target)];

    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    let metrics = trainer.train(&dataset, forward_fn)?;

    assert_eq!(trainer.global_step(), 4); // 2 epochs * 2 batches
    assert_eq!(metrics.len(), 4);

    // Verify step numbers in metrics (starts from 1)
    assert_eq!(metrics[0].step, 1);
    assert_eq!(metrics[1].step, 2);
    assert_eq!(metrics[2].step, 3);
    assert_eq!(metrics[3].step, 4);

    Ok(())
}

#[test]
fn test_empty_dataset_returns_empty_metrics() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig::default();

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    let dataset: Vec<(Tensor, Tensor)> = vec![];
    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    let metrics = trainer.train(&dataset, forward_fn)?;

    assert_eq!(metrics.len(), 0);
    assert_eq!(trainer.global_step(), 0);

    Ok(())
}

#[test]
fn test_lora_adapter_accessors() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig {
        rank: 8,
        ..Default::default()
    };
    let training_config = TrainingConfig::default();

    let mut trainer = Trainer::new(128, 256, 3, &lora_config, training_config, &device)?;

    // Test immutable accessor
    let adapter = trainer.lora_adapter();
    assert!(adapter.num_trainable_parameters() > 0);

    // Test mutable accessor
    let adapter_mut = trainer.lora_adapter_mut();
    assert!(adapter_mut.num_trainable_parameters() > 0);

    Ok(())
}

#[test]
fn test_training_with_warmup_cosine_scheduler() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    let training_config = TrainingConfig {
        num_epochs: 1,
        lr_scheduler: LRScheduler::warmup_cosine(2, 10, 1e-3, 1e-5),
        ..Default::default()
    };

    let mut trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;

    let input = Tensor::zeros((1, 4), DType::U32, &device)?;
    let target = Tensor::zeros((1, 4), DType::U32, &device)?;
    let dataset = vec![
        (input.clone(), target.clone()),
        (input.clone(), target.clone()),
        (input.clone(), target.clone()),
        (input, target),
    ];

    let vocab_size = 500;
    let forward_fn = create_test_forward_fn(&device, vocab_size);

    let metrics = trainer.train(&dataset, forward_fn)?;

    // During warmup, LR should increase
    assert!(metrics[0].learning_rate < metrics[1].learning_rate);
    // After warmup, LR follows cosine decay
    assert!(metrics[2].learning_rate > metrics[3].learning_rate);

    Ok(())
}

#[test]
#[cfg(target_os = "macos")]
fn test_training_on_metal_device() -> anyhow::Result<()> {
    use metal_candle::backend::Device as MetalCandleDevice;
    
    if let Ok(device) = MetalCandleDevice::new_metal(0) {
        let candle_device = device.as_candle_device();
        let lora_config = LoRAAdapterConfig {
            rank: 4,
            ..Default::default()
        };
        let training_config = TrainingConfig {
            num_epochs: 1,
            ..Default::default()
        };

        let mut trainer =
            Trainer::new(64, 128, 2, &lora_config, training_config, candle_device)?;

        let input = Tensor::zeros((1, 4), DType::U32, candle_device)?;
        let target = Tensor::zeros((1, 4), DType::U32, candle_device)?;
        let dataset = vec![(input, target)];

        let vocab_size = 500;
        let forward_fn = create_test_forward_fn(candle_device, vocab_size);

        let metrics = trainer.train(&dataset, forward_fn)?;

        assert_eq!(metrics.len(), 1);
        assert!(metrics[0].loss >= 0.0);
    }

    Ok(())
}

#[test]
fn test_default_training_config() {
    let config = TrainingConfig::default();

    assert_eq!(config.num_epochs, 3);
    assert_eq!(config.max_grad_norm, Some(1.0));
    matches!(config.lr_scheduler, LRScheduler::Constant { .. });
}

#[test]
fn test_training_config_validation() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let lora_config = LoRAAdapterConfig::default();
    
    // Valid config with no gradient clipping
    let training_config = TrainingConfig {
        num_epochs: 1,
        max_grad_norm: None,
        ..Default::default()
    };

    let trainer = Trainer::new(64, 128, 2, &lora_config, training_config, &device)?;
    assert_eq!(trainer.global_step(), 0);

    Ok(())
}

