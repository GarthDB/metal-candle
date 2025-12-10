//! Baseline benchmark for Candle's Metal matmul.

use candle_core::Tensor;
use metal_candle::backend::Device;
use std::time::Instant;

fn main() {
    println!("=== Candle Metal Matmul Baseline ===\n");

    let device = Device::new_metal(0).expect("Metal device");
    let candle_device = device.as_candle_device();

    let sizes = vec![
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    println!(
        "{:<15} {:<15} {:<15}",
        "Size", "Time (µs)", "Throughput (GFLOPS)"
    );
    println!("{}", "-".repeat(50));

    for (m, k, n) in sizes {
        let a = Tensor::randn(0.0f32, 1.0, (m, k), candle_device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (k, n), candle_device).unwrap();

        // Warmup
        let _ = a.matmul(&b).unwrap();

        // Benchmark
        let iterations = 1000; // More iterations for stable measurement
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a.matmul(&b).unwrap();
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;

        // Calculate GFLOPS: 2*M*N*K operations
        let ops = 2.0 * m as f64 * n as f64 * k as f64;
        let gflops = (ops / (avg_us * 1000.0)); // Convert µs to seconds, ops to GFLOPS

        println!(
            "{:<15} {:<15.2} {:<15.2}",
            format!("{}x{}x{}", m, k, n),
            avg_us,
            gflops
        );
    }
}
