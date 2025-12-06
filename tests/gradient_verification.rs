//! Gradient verification tests for `LoRA` training.
//!
//! These tests verify that:
//! 1. Gradients flow correctly through `LoRA` layers
//! 2. Parameter updates work as expected
//! 3. Training steps produce valid gradients
//! 4. `LoRA` adapter integrates properly with autograd

use candle_core::{DType, Device, Tensor};
use metal_candle::training::{
    AdamW, AdamWConfig, LoRAAdapter, LoRAAdapterConfig, LoRAConfig, LoRALayer, TargetModule,
    TrainingStep,
};

#[test]
fn test_lora_layer_gradient_flow() {
    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(16, 16, &config, &device).unwrap();

    // Manually initialize B with non-zero values for testing
    // (In real training, B starts at zero but gets updated)
    // NOTE: LoRA matrices are stored in transposed form for optimization
    // lora_b is stored as (rank, out_features) instead of (out_features, rank)
    let b_init = Tensor::randn(0f32, 0.1, (4, 16), &device).unwrap();
    lora.lora_b().set(&b_init).unwrap();

    // Create input
    let input = Tensor::randn(0f32, 1f32, (2, 16), &device).unwrap();

    // Forward pass
    let output = lora.forward(&input).unwrap();

    // Compute MSE loss with a target
    let target = Tensor::randn(0f32, 1f32, (2, 16), &device).unwrap();
    let diff = (output - target).unwrap();
    let loss = diff.sqr().unwrap().mean_all().unwrap();

    // Backward pass
    let grads = loss.backward().unwrap();

    // Verify gradients exist
    assert!(
        grads.get(lora.lora_a()).is_some(),
        "Gradient for A matrix should exist"
    );
    assert!(
        grads.get(lora.lora_b()).is_some(),
        "Gradient for B matrix should exist"
    );

    // Verify gradient shapes match parameter shapes
    let grad_a = grads.get(lora.lora_a()).unwrap();
    let grad_b = grads.get(lora.lora_b()).unwrap();

    assert_eq!(grad_a.dims(), lora.lora_a_tensor().dims());
    assert_eq!(grad_b.dims(), lora.lora_b_tensor().dims());

    // Verify gradients are non-zero (learning signal exists)
    let grad_a_sum = grad_a
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();
    let grad_b_total = grad_b
        .abs()
        .unwrap()
        .sum_all()
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();

    assert!(
        grad_a_sum > 1e-6,
        "Gradient A should be non-zero, got {grad_a_sum}"
    );
    assert!(
        grad_b_total > 1e-6,
        "Gradient B should be non-zero, got {grad_b_total}"
    );
}

#[test]
fn test_lora_adapter_gradient_collection() {
    let device = Device::Cpu;
    let adapter_config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };

    let lora_adapter = LoRAAdapter::new(32, 128, 2, &adapter_config, &device).unwrap();

    // Get all trainable variables
    let mut all_vars = Vec::new();
    for (_key, layer) in lora_adapter.layers() {
        all_vars.extend(layer.trainable_variables());
    }

    // Should have 2 layers × 2 modules × 2 matrices = 8 vars
    assert_eq!(all_vars.len(), 8, "Should have 8 trainable variables");

    // Verify we can compute gradients for a simple forward pass
    // Create dummy input for q_proj (layer 0)
    let input = Tensor::randn(0f32, 1f32, (1, 8, 32), &device).unwrap();

    // Apply LoRA delta
    let delta = lora_adapter
        .forward(0, &TargetModule::QProj, &input)
        .unwrap();

    if let Some(delta_tensor) = delta {
        let loss = delta_tensor.sum_all().unwrap();
        let grads = loss.backward().unwrap();

        // Check that at least some variables have gradients
        let vars_with_grads = all_vars.iter().filter(|v| grads.get(v).is_some()).count();
        assert!(
            vars_with_grads > 0,
            "At least some LoRA parameters should have gradients"
        );
    }
}

#[test]
fn test_optimizer_updates_parameters() {
    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(16, 16, &config, &device).unwrap();

    // Store initial values
    let initial_a = lora.lora_a_tensor().to_vec2::<f32>().unwrap();

    // Create optimizer
    let opt_config = AdamWConfig::default();
    let mut optimizer = AdamW::new(opt_config).unwrap();

    // Perform forward pass and compute gradients
    let input = Tensor::ones((2, 16), DType::F32, &device).unwrap();
    let output = lora.forward(&input).unwrap();
    let loss = output.sum_all().unwrap();
    let grads = loss.backward().unwrap();

    // Update parameters
    let grad_a = grads.get(lora.lora_a()).unwrap();
    optimizer.step_var(lora.lora_a(), grad_a).unwrap();

    // Verify parameters changed
    let updated_a = lora.lora_a_tensor().to_vec2::<f32>().unwrap();

    // Check that at least one element changed
    let mut changed = false;
    for (i, row) in initial_a.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if (val - updated_a[i][j]).abs() > 1e-6 {
                changed = true;
                break;
            }
        }
        if changed {
            break;
        }
    }

    assert!(changed, "Parameters should change after optimizer step");
}

#[test]
fn test_training_step_produces_valid_gradients() {
    let device = Device::Cpu;

    let adapter_config = LoRAAdapterConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj],
    };

    let lora_adapter = LoRAAdapter::new(32, 128, 1, &adapter_config, &device).unwrap();

    let opt_config = AdamWConfig::default();
    let mut optimizer = AdamW::new(opt_config).unwrap();

    let mut step = TrainingStep::new();

    // Prepare dummy data
    let input_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();
    let target_ids = Tensor::zeros((1, 8), DType::U32, &device).unwrap();

    // Define forward function
    let forward_fn = |_input: &Tensor| -> metal_candle::Result<Tensor> {
        Ok(Tensor::randn(0f32, 1f32, (1, 8, 100), &device)?)
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

    // Verify loss is valid
    assert!(metrics.loss.is_finite(), "Loss should be finite");
    assert!(metrics.loss >= 0.0, "Loss should be non-negative");
}

#[test]
fn test_gradient_accumulation_over_steps() {
    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(16, 16, &config, &device).unwrap();

    // Initialize B with non-zero values for testing
    // NOTE: LoRA matrices are stored in transposed form for optimization
    // lora_b is stored as (rank, out_features) instead of (out_features, rank)
    let b_init = Tensor::randn(0f32, 0.1, (4, 16), &device).unwrap();
    lora.lora_b().set(&b_init).unwrap();

    let opt_config = AdamWConfig {
        learning_rate: 0.01, // Higher LR for visible changes
        ..Default::default()
    };
    let mut optimizer = AdamW::new(opt_config).unwrap();

    // Store initial parameters (as actual values, not reference)
    let initial_a_values = lora.lora_a_tensor().to_vec2::<f32>().unwrap();

    // Perform multiple training steps
    for _ in 0..5 {
        let input = Tensor::randn(0f32, 1f32, (2, 16), &device).unwrap();
        let output = lora.forward(&input).unwrap();
        let target = Tensor::randn(0f32, 1f32, (2, 16), &device).unwrap();
        let diff = (output - target).unwrap();
        let loss = diff.sqr().unwrap().mean_all().unwrap();
        let grads = loss.backward().unwrap();

        let grad_a = grads.get(lora.lora_a()).unwrap();
        let grad_b = grads.get(lora.lora_b()).unwrap();
        optimizer.step_var(lora.lora_a(), grad_a).unwrap();
        optimizer.step_var(lora.lora_b(), grad_b).unwrap();
    }

    // Verify parameters have changed significantly
    let final_a_values = lora.lora_a_tensor().to_vec2::<f32>().unwrap();

    // Calculate total absolute difference
    let mut diff_value = 0.0f32;
    for (i, row) in initial_a_values.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            diff_value += (val - final_a_values[i][j]).abs();
        }
    }

    assert!(
        diff_value > 1e-6,
        "Parameters should change significantly over multiple steps, diff: {diff_value}"
    );
}

#[test]
fn test_zero_gradients_with_frozen_params() {
    // This test verifies that if we don't include a parameter in the backward pass,
    // it doesn't get gradients (simulating frozen base model parameters)

    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(16, 16, &config, &device).unwrap();

    // Create a separate tensor that won't be part of the computation graph
    let frozen_param = Tensor::randn(0f32, 1f32, (16, 16), &device).unwrap();
    let frozen_var = candle_core::Var::from_tensor(&frozen_param).unwrap();

    // Forward pass (doesn't use frozen_var)
    let input = Tensor::ones((2, 16), DType::F32, &device).unwrap();
    let output = lora.forward(&input).unwrap();
    let loss = output.sum_all().unwrap();

    // Backward pass
    let grads = loss.backward().unwrap();

    // LoRA params should have gradients
    assert!(
        grads.get(lora.lora_a()).is_some(),
        "LoRA A should have gradients"
    );
    assert!(
        grads.get(lora.lora_b()).is_some(),
        "LoRA B should have gradients"
    );

    // Frozen param should NOT have gradients
    assert!(
        grads.get(&frozen_var).is_none(),
        "Frozen parameter should not have gradients"
    );
}
