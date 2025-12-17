//! Simple manual benchmark for MPS vs Candle matmul.
//!
//! Avoids Criterion's heavy iteration to prevent GPU memory exhaustion.

use candle_core::Tensor;
use metal_candle::backend::Device;
use std::time::Instant;

#[cfg(feature = "mps")]
use metal_candle::backend::mps::mps_matmul;

fn main() {
    println!("=== MPS Matrix Multiplication Benchmark ===\n");

    let device = Device::new_metal(0).expect("Metal device");
    let candle_device = device.as_candle_device();

    // Test various sizes
    let sizes = vec![
        (64, 64, 64, "Tiny"),
        (128, 128, 128, "Small"),
        (256, 256, 256, "Medium"),
        (512, 512, 512, "Large"),
    ];

    println!("Format: (M, K, N) - M×K @ K×N = M×N\n");
    println!(
        "{:<15} {:<15} {:<15} {:<10}",
        "Size", "Candle (µs)", "MPS (µs)", "Speedup"
    );
    println!("{}", "-".repeat(60));

    for (m, k, n, label) in sizes {
        // Create test matrices
        let a = Tensor::randn(0.0f32, 1.0, (m, k), candle_device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, (k, n), candle_device).unwrap();

        // Warmup (1 iteration each)
        let _ = a.matmul(&b).unwrap();
        #[cfg(feature = "mps")]
        let _ = mps_matmul(&a, &b).unwrap();

        // Benchmark Candle matmul
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = a.matmul(&b).unwrap();
        }
        let candle_duration = start.elapsed();
        let candle_avg_us = candle_duration.as_micros() as f64 / iterations as f64;

        // Benchmark MPS matmul
        #[cfg(feature = "mps")]
        let (mps_avg_us, speedup) = {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = mps_matmul(&a, &b).unwrap();
            }
            let mps_duration = start.elapsed();
            let mps_avg = mps_duration.as_micros() as f64 / iterations as f64;
            let speedup = candle_avg_us / mps_avg;
            (mps_avg, speedup)
        };

        #[cfg(not(feature = "mps"))]
        let (mps_avg_us, speedup) = (0.0, 0.0);

        println!(
            "{:<15} {:<15.2} {:<15.2} {:<10.2}x",
            format!("{}x{}x{}", m, k, n),
            candle_avg_us,
            mps_avg_us,
            speedup
        );
    }

    println!("\n=== LoRA-Specific Sizes ===\n");
    println!(
        "{:<20} {:<15} {:<15} {:<10}",
        "Config", "Candle (µs)", "MPS (µs)", "Speedup"
    );
    println!("{}", "-".repeat(65));

    let lora_configs = vec![
        (32, 512, 8, "bs32, dim512, r8"),
        (64, 1024, 16, "bs64, dim1024, r16"),
        (128, 2048, 32, "bs128, dim2048, r32"),
    ];

    for (batch_seq, dim, rank, label) in lora_configs {
        let input = Tensor::randn(0.0f32, 1.0, (batch_seq, dim), candle_device).unwrap();
        let lora_a = Tensor::randn(0.0f32, 0.01, (dim, rank), candle_device).unwrap();

        // Warmup
        let _ = input.matmul(&lora_a).unwrap();
        #[cfg(feature = "mps")]
        let _ = mps_matmul(&input, &lora_a).unwrap();

        // Benchmark
        let iterations = 100;

        let start = Instant::now();
        for _ in 0..iterations {
            let _ = input.matmul(&lora_a).unwrap();
        }
        let candle_avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        #[cfg(feature = "mps")]
        let (mps_avg_us, speedup) = {
            let start = Instant::now();
            for _ in 0..iterations {
                let _ = mps_matmul(&input, &lora_a).unwrap();
            }
            let mps_avg = start.elapsed().as_micros() as f64 / iterations as f64;
            (mps_avg, candle_avg_us / mps_avg)
        };

        #[cfg(not(feature = "mps"))]
        let (mps_avg_us, speedup) = (0.0, 0.0);

        println!(
            "{:<20} {:<15.2} {:<15.2} {:<10.2}x",
            label, candle_avg_us, mps_avg_us, speedup
        );
    }

    println!("\nNote: Times are averaged over {} iterations", 100);
}


