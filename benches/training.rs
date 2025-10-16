//! Training benchmarks for metal-candle.
//!
//! Measures `LoRA` training performance including:
//! - Forward pass
//! - Backward pass (gradient computation)
//! - Optimizer step
//! - Full training iteration
//!
//! Run with: `cargo bench --bench training`

#![allow(missing_docs)]

use candle_core::{DType, Device, Tensor, Var};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use metal_candle::{
    backend::TensorExt,
    training::{cross_entropy_loss, AdamW, AdamWConfig, LoRAConfig, LoRALayer},
    Device as MetalDevice,
};

fn benchmark_lora_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_forward");

    let _device = MetalDevice::new_with_fallback(0);
    let candle_device = Device::Cpu; // Use CPU for consistent benchmarking

    // Different dimensions to benchmark
    let configs = vec![
        ("small_512x512_r8", 512, 512, 8),
        ("medium_1024x1024_r8", 1024, 1024, 8),
        ("large_2048x2048_r8", 2048, 2048, 8),
        ("small_512x512_r16", 512, 512, 16),
    ];

    for (name, in_features, out_features, rank) in configs {
        let alpha = match rank {
            4 => 8.0,
            8 => 16.0,
            16 => 32.0,
            32 => 64.0,
            64 => 128.0,
            _ => 16.0, // Default fallback
        };
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let lora = LoRALayer::new(in_features, out_features, &config, &candle_device)
            .expect("Failed to create LoRA layer");

        let input = Tensor::randn(0f32, 1f32, (4, 16, in_features), &candle_device)
            .expect("Failed to create input");

        group.throughput(Throughput::Elements((4 * 16 * in_features) as u64));

        group.bench_with_input(
            BenchmarkId::new("forward", name),
            &(&lora, &input),
            |b, (lora, input)| {
                b.iter(|| {
                    let output = lora.forward(black_box(input)).expect("Forward failed");
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_gradient_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_computation");

    let device = Device::Cpu;
    let config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(512, 512, &config, &device).expect("Failed to create LoRA layer");
    let input = Tensor::randn(0f32, 1f32, (2, 16, 512), &device).expect("Failed to create input");
    let labels = Tensor::zeros((2, 16), DType::U32, &device).expect("Failed to create labels");

    group.bench_function("forward_backward", |b| {
        b.iter(|| {
            // Forward pass
            let output = lora.forward(&input).expect("Forward failed");

            // Compute loss
            let loss = cross_entropy_loss(&output, &labels, None).expect("Loss failed");

            // Backward pass
            let grads = loss.backward().expect("Backward failed");

            black_box(grads)
        });
    });

    group.finish();
}

fn benchmark_optimizer_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimizer_step");

    let device = Device::Cpu;
    let config = AdamWConfig::default();
    let mut optimizer = AdamW::new(config).expect("Failed to create optimizer");

    // Create dummy parameter
    let param_data =
        Tensor::randn(0f32, 1f32, (512, 512), &device).expect("Failed to create param");
    let param = Var::from_tensor(&param_data).expect("Failed to create Var");

    let grad = Tensor::randn(0f32, 1f32, (512, 512), &device).expect("Failed to create grad");

    group.bench_function("adamw_step", |b| {
        b.iter(|| {
            optimizer
                .step_var(black_box(&param), black_box(&grad))
                .expect("Optimizer step failed");
        });
    });

    group.finish();
}

fn benchmark_full_training_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_training_step");
    group.sample_size(10); // Fewer samples for expensive operation

    let device = Device::Cpu;

    // Setup LoRA layer
    let lora_config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(512, 512, &lora_config, &device).expect("Failed to create LoRA");

    // Create optimizer
    let opt_config = AdamWConfig::default();
    let mut optimizer = AdamW::new(opt_config).expect("Failed to create optimizer");

    // Create dummy data
    let input = Tensor::randn(0f32, 1f32, (4, 64, 512), &device).expect("Failed to create input");
    let labels = Tensor::zeros((4, 64), DType::U32, &device).expect("Failed to create labels");

    group.throughput(Throughput::Elements((4 * 64 * 512) as u64));

    group.bench_function("complete_iteration", |b| {
        b.iter(|| {
            // Forward
            let output = lora.forward(&input).expect("Forward failed");

            // Loss
            let loss = cross_entropy_loss(&output, &labels, None).expect("Loss failed");

            // Backward
            let grads = loss.backward().expect("Backward failed");

            // Extract gradients
            if let (Some(grad_a), Some(grad_b)) =
                (grads.get(lora.lora_a()), grads.get(lora.lora_b()))
            {
                // Optimizer step
                optimizer
                    .step_var(lora.lora_a(), grad_a)
                    .expect("Step A failed");
                optimizer
                    .step_var(lora.lora_b(), grad_b)
                    .expect("Step B failed");
            }

            black_box(loss)
        });
    });

    group.finish();
}

fn benchmark_layer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("layer_operations");

    let device = Device::Cpu;
    let size = 1024;

    let tensor =
        Tensor::randn(0f32, 1f32, (4, 16, size), &device).expect("Failed to create tensor");

    // Softmax
    group.bench_function("softmax_stable", |b| {
        b.iter(|| {
            let result = tensor.softmax_stable().expect("Softmax failed");
            black_box(result)
        });
    });

    // Layer norm
    group.bench_function("layer_norm", |b| {
        b.iter(|| {
            let result = tensor.layer_norm(1e-5).expect("Layer norm failed");
            black_box(result)
        });
    });

    // RMS norm
    group.bench_function("rms_norm", |b| {
        b.iter(|| {
            let result = tensor.rms_norm(1e-5).expect("RMS norm failed");
            black_box(result)
        });
    });

    group.finish();
}

fn benchmark_lora_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_rank_scaling");

    let device = Device::Cpu;
    let in_features = 1024;
    let out_features = 1024;

    for rank in [4, 8, 16, 32, 64] {
        let alpha = match rank {
            4 => 8.0,
            8 => 16.0,
            16 => 32.0,
            32 => 64.0,
            64 => 128.0,
            _ => 16.0, // Default fallback
        };
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let lora = LoRALayer::new(in_features, out_features, &config, &device)
            .expect("Failed to create LoRA layer");

        let input = Tensor::randn(0f32, 1f32, (2, 32, in_features), &device)
            .expect("Failed to create input");

        group.bench_with_input(
            BenchmarkId::new("forward", format!("rank_{rank}")),
            &(&lora, &input),
            |b, (lora, input)| {
                b.iter(|| {
                    let output = lora.forward(black_box(input)).expect("Forward failed");
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    training_benches,
    benchmark_lora_forward,
    benchmark_gradient_computation,
    benchmark_optimizer_step,
    benchmark_full_training_step,
    benchmark_layer_operations,
    benchmark_lora_scaling,
);
criterion_main!(training_benches);
