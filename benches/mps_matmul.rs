//! Benchmark MPS matrix multiplication vs custom kernels and Candle.

use candle_core::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use metal_candle::backend::Device;

// MPS matmul will be called via mps_matmul function, not trait

/// Benchmark MPS matmul vs Candle matmul for various sizes.
fn bench_mps_vs_candle(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_comparison");

    // Test various matrix sizes (start smaller to avoid OOM)
    let sizes = vec![
        (64, 64, 64),    // Tiny
        (128, 128, 128), // Small
        (256, 256, 256), // Medium
    ];

    for (m, k, n) in sizes {
        let size_str = format!("{}x{}x{}", m, k, n);

        // Setup
        let device = Device::new_metal(0).expect("Metal device");
        let candle_device = device.as_candle_device();

        let a = Tensor::randn(0.0f32, 1.0, (m, k), candle_device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (k, n), candle_device).unwrap();

        // Benchmark Candle's matmul
        group.bench_with_input(
            BenchmarkId::new("candle_matmul", &size_str),
            &(&a, &b),
            |bencher, (a, b)| {
                bencher.iter(|| black_box(a.matmul(b).unwrap()));
            },
        );

        // Benchmark MPS matmul
        #[cfg(feature = "mps")]
        {
            use metal_candle::backend::mps::mps_matmul;
            group.bench_with_input(
                BenchmarkId::new("mps_matmul", &size_str),
                &(&a, &b),
                |bencher, (a, b)| {
                    bencher.iter(|| black_box(mps_matmul(a, b).unwrap()));
                },
            );
        }
    }

    group.finish();
}

/// Benchmark LoRA-specific matmul sizes.
fn bench_lora_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora_matmul");

    // LoRA-specific sizes: (batch*seq, in_features) @ (in_features, rank)
    let lora_sizes = vec![
        (16, 256, 8), // Tiny LoRA
        (32, 512, 8), // Small LoRA
    ];

    for (batch_seq, in_feat, rank) in lora_sizes {
        let size_str = format!("bs{}x{}xr{}", batch_seq, in_feat, rank);

        let device = Device::new_metal(0).expect("Metal device");
        let candle_device = device.as_candle_device();

        let input = Tensor::randn(0.0f32, 1.0, (batch_seq, in_feat), candle_device).unwrap();
        let lora_a = Tensor::randn(0.0f32, 0.01, (in_feat, rank), candle_device).unwrap();

        // Candle
        group.bench_with_input(
            BenchmarkId::new("candle", &size_str),
            &(&input, &lora_a),
            |bencher, (input, lora_a)| {
                bencher.iter(|| black_box(input.matmul(lora_a).unwrap()));
            },
        );

        // MPS
        #[cfg(feature = "mps")]
        {
            use metal_candle::backend::mps::mps_matmul;
            group.bench_with_input(
                BenchmarkId::new("mps", &size_str),
                &(&input, &lora_a),
                |bencher, (input, lora_a)| {
                    bencher.iter(|| black_box(mps_matmul(input, lora_a).unwrap()));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_mps_vs_candle, bench_lora_sizes);
criterion_main!(benches);
