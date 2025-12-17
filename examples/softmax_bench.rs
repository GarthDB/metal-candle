//! Softmax Performance Benchmark
//!
//! Compares performance of fused softmax kernel vs Candle's default implementation.
//!
//! Run with: `cargo run --release --example softmax_bench --features custom-metal`

use candle_core::{Device, Tensor, D};
use metal_candle::backend::metal_ops::CustomMetalOps;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 Softmax Performance Benchmark");
    println!("═══════════════════════════════════════════════════════\n");

    let device = Device::new_metal(0)?;

    // Test configuration matching MLX baseline
    let batch_size = 1;
    let seq_len = 128;
    let dim = 1024;
    let iterations = 1000;

    println!("Configuration:");
    println!("  Batch size: {batch_size}");
    println!("  Sequence length: {seq_len}");
    println!("  Dimension: {dim}");
    println!("  Iterations: {iterations}");

    // Create test input
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, dim), &device)?;

    println!("\n📊 Benchmarking Fused Softmax...");

    // Warmup
    for _ in 0..10 {
        let _ = input.softmax_fused()?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = input.softmax_fused()?;
    }
    let fused_duration = start.elapsed();
    let fused_time_us = fused_duration.as_micros() as f64 / iterations as f64;

    println!("  Fused softmax: {:.2} µs", fused_time_us);

    println!("\n📊 Benchmarking Candle Softmax...");

    // Warmup
    for _ in 0..10 {
        let _ = candle_nn::ops::softmax(&input, D::Minus1)?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = candle_nn::ops::softmax(&input, D::Minus1)?;
    }
    let candle_duration = start.elapsed();
    let candle_time_us = candle_duration.as_micros() as f64 / iterations as f64;

    println!("  Candle softmax: {:.2} µs", candle_time_us);

    println!("\n🎯 Results:");
    println!("═══════════════════════════════════════════════════════");
    println!("  Fused:  {:.2} µs", fused_time_us);
    println!("  Candle: {:.2} µs", candle_time_us);

    let speedup = candle_time_us / fused_time_us;
    println!("  Speedup: {:.2}x", speedup);

    println!("\n  Comparison:");
    println!("    MLX baseline:  1.85 µs");
    println!("    Candle (before): {:.2} µs", candle_time_us);
    println!("    Our fused:     {:.2} µs", fused_time_us);

    if fused_time_us < 7.0 {
        println!("\n✅ EXCELLENT: Achieved target performance (< 7 µs)!");
    } else if fused_time_us < candle_time_us * 0.5 {
        println!("\n✅ GOOD: {:.2}x speedup achieved!", speedup);
    } else {
        println!("\n⚠️  MODEST: Only {:.2}x speedup", speedup);
    }

    // Test multiple dimensions
    println!("\n📊 Testing Various Dimensions:");
    println!("═══════════════════════════════════════════════════════");

    let test_dims = vec![256, 512, 1024, 2048, 4096];

    for &d in &test_dims {
        let test_input = Tensor::randn(0f32, 1f32, (1, 128, d), &device)?;

        // Warmup
        for _ in 0..5 {
            let _ = test_input.softmax_fused()?;
        }

        let start = Instant::now();
        for _ in 0..100 {
            let _ = test_input.softmax_fused()?;
        }
        let time_us = start.elapsed().as_micros() as f64 / 100.0;

        println!("  Dim {d:4}: {:.2} µs", time_us);
    }

    Ok(())
}
