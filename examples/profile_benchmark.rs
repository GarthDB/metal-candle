//! Profiling benchmark for Xcode Instruments
//!
//! This example runs the same operations as our MLX comparison
//! but in a way that's easy to profile with Instruments.
//!
//! Run with: cargo instruments -t "Metal System Trace" --release --example profile_benchmark

use candle_core::{DType, Device, Tensor};
use metal_candle::{
    backend::TensorExt,
    training::{LoRAConfig, LoRALayer},
};

fn main() -> anyhow::Result<()> {
    println!("🔍 Profiling Benchmark for Xcode Instruments");
    println!("═══════════════════════════════════════════════");
    println!();

    let device = Device::new_metal(0)?;
    println!("✅ Device: Metal GPU");
    println!();

    // Warmup GPU
    println!("Warming up GPU...");
    let warmup = Tensor::zeros((1024, 1024), DType::F32, &device)?;
    let _ = warmup.matmul(&warmup)?;
    println!("✅ GPU warmed up");
    println!();

    // === LoRA Forward Pass Benchmarks ===
    println!("1️⃣  LoRA Forward Pass Benchmarks");
    println!("───────────────────────────────────────────────");

    for (name, in_features, out_features, rank) in [
        ("small_512x512_r8", 512, 512, 8),
        ("medium_1024x1024_r8", 1024, 1024, 8),
        ("large_2048x2048_r8", 2048, 2048, 8),
    ] {
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let layer = LoRALayer::new(in_features, out_features, &config, &device)?;
        let x = Tensor::randn(0f32, 1f32, (1, in_features), &device)?;

        println!("  Running: {} ({}x{}, rank={})", name, in_features, out_features, rank);

        // Run 1000 iterations for profiling
        for _ in 0..1000 {
            let _ = layer.forward(&x)?;
        }
    }
    println!("✅ LoRA benchmarks complete");
    println!();

    // === Layer Operations Benchmarks ===
    println!("2️⃣  Layer Operations Benchmarks");
    println!("───────────────────────────────────────────────");

    let size = 1024;
    let tensor = Tensor::randn(0f32, 1f32, (4, 16, size), &device)?;

    // Softmax
    println!("  Running: softmax_stable");
    for _ in 0..1000 {
        let _ = tensor.softmax_stable()?;
    }

    // Layer Norm
    println!("  Running: layer_norm");
    for _ in 0..1000 {
        let _ = tensor.layer_norm(1e-5)?;
    }

    // RMS Norm
    println!("  Running: rms_norm");
    for _ in 0..1000 {
        let _ = tensor.rms_norm(1e-5)?;
    }
    println!("✅ Layer operations complete");
    println!();

    // === LoRA Rank Scaling ===
    println!("3️⃣  LoRA Rank Scaling");
    println!("───────────────────────────────────────────────");

    let in_features = 1024;
    let out_features = 1024;
    for rank in [4, 8, 16, 32, 64] {
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let layer = LoRALayer::new(in_features, out_features, &config, &device)?;
        let x = Tensor::randn(0f32, 1f32, (1, in_features), &device)?;

        println!("  Running: rank_{}", rank);
        for _ in 0..1000 {
            let _ = layer.forward(&x)?;
        }
    }
    println!("✅ Rank scaling complete");
    println!();

    println!("═══════════════════════════════════════════════");
    println!("✅ Profiling benchmark complete!");
    println!();
    println!("Analyze results in Xcode Instruments:");
    println!("  - Check Metal System Trace");
    println!("  - Look at GPU timeline");
    println!("  - Identify kernel launch overhead");
    println!("  - Check memory transfers");
    println!();

    Ok(())
}

