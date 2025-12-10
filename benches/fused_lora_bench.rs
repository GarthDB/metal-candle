//! Benchmark for fused LoRA kernel vs unfused Candle operations.

#![allow(missing_docs)]

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use metal_candle::backend::custom_ops::FusedLoRAOp;

fn bench_fused_vs_unfused(c: &mut Criterion) {
    let device = Device::new_metal(0).expect("Metal device required for benchmark");

    // Test configuration: typical LoRA parameters
    let batch_size = 1;
    let seq_len = 128;
    let features = 512;
    let rank = 8;
    let scaling = 2.0f32;

    // Create test tensors
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, features), &device).unwrap();
    let lora_a = Tensor::randn(0f32, 0.01f32, (features, rank), &device).unwrap();
    let lora_b = Tensor::randn(0f32, 0.01f32, (rank, features), &device).unwrap();

    // Benchmark unfused (standard Candle operations)
    c.bench_function("lora_unfused (Candle)", |b| {
        b.iter(|| {
            let hidden = black_box(&input).broadcast_matmul(&lora_a).unwrap();
            let output = hidden.broadcast_matmul(&lora_b).unwrap();
            let scaled = output.affine(f64::from(scaling), 0.0).unwrap();
            black_box(scaled)
        });
    });

    // Benchmark fused (custom Metal kernel)
    c.bench_function("lora_fused (Metal)", |b| {
        b.iter(|| {
            let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), scaling).unwrap();
            let output = black_box(&input).apply_op1(op).unwrap();
            black_box(output)
        });
    });
}

criterion_group!(benches, bench_fused_vs_unfused);
criterion_main!(benches);
