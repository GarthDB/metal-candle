//! RMS Norm Performance Benchmark
//!
//! Compares performance of fused RMS norm kernel vs Candle's default implementation.
//!
//! Run with: `cargo run --release --example rmsnorm_bench --features custom-metal`

use candle_core::{Device, Tensor};
use metal_candle::backend::metal_ops::CustomMetalOps;
use std::time::Instant;

/// Reference RMS norm implementation using Candle ops
fn rms_norm_reference(x: &Tensor, eps: f32) -> Result<Tensor, candle_core::Error> {
    let x_sq = x.sqr()?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;
    let rms = (mean_sq + eps as f64)?.sqrt()?;
    x.broadcast_div(&rms)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🚀 RMS Norm Performance Benchmark");
    println!("═══════════════════════════════════════════════════════\n");

    let device = Device::new_metal(0)?;

    // Test configuration matching MLX baseline
    let batch_size = 1;
    let seq_len = 128;
    let dim = 1024;
    let eps = 1e-5f32;
    let iterations = 1000;

    println!("Configuration:");
    println!("  Batch size: {batch_size}");
    println!("  Sequence length: {seq_len}");
    println!("  Dimension: {dim}");
    println!("  Epsilon: {eps}");
    println!("  Iterations: {iterations}");

    // Create test input
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, dim), &device)?;

    println!("\n📊 Benchmarking Fused RMS Norm...");

    // Warmup
    for _ in 0..10 {
        let _ = input.rms_norm_fused(eps)?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = input.rms_norm_fused(eps)?;
    }
    let fused_duration = start.elapsed();
    let fused_time_us = fused_duration.as_micros() as f64 / iterations as f64;

    println!("  Fused RMS norm: {:.2} µs", fused_time_us);

    println!("\n📊 Benchmarking Candle RMS Norm...");

    // Warmup
    for _ in 0..10 {
        let _ = rms_norm_reference(&input, eps)?;
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = rms_norm_reference(&input, eps)?;
    }
    let candle_duration = start.elapsed();
    let candle_time_us = candle_duration.as_micros() as f64 / iterations as f64;

    println!("  Candle RMS norm: {:.2} µs", candle_time_us);

    println!("\n🎯 Results:");
    println!("═══════════════════════════════════════════════════════");
    println!("  Fused:  {:.2} µs", fused_time_us);
    println!("  Candle: {:.2} µs", candle_time_us);

    let speedup = candle_time_us / fused_time_us;
    println!("  Speedup: {:.2}x", speedup);

    println!("\n  Comparison:");
    println!("    MLX baseline:  6.08 µs");
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
            let _ = test_input.rms_norm_fused(eps)?;
        }

        let start = Instant::now();
        for _ in 0..100 {
            let _ = test_input.rms_norm_fused(eps)?;
        }
        let time_us = start.elapsed().as_micros() as f64 / 100.0;

        println!("  Dim {d:4}: {:.2} µs", time_us);
    }

    Ok(())
}
