//! Benchmark LoRALayer with automatic fused kernel selection.

use candle_core::{Device, Tensor};
use metal_candle::training::{LoRAConfig, LoRALayer};
use std::time::Instant;

fn main() {
    println!("\n🚀 LoRALayer Performance Test (Automatic Fused Kernel)");
    println!("═══════════════════════════════════════════════════════");

    let device = Device::new_metal(0).expect("Metal device required");

    // Test configuration
    let batch_size = 1;
    let seq_len = 128;
    let in_features = 512;
    let out_features = 512;
    let rank = 8;
    let alpha = 16.0;
    let iterations = 100;
    let warmup = 10;

    println!("\nConfiguration:");
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Features: {}", in_features);
    println!("  Rank: {}", rank);
    println!("  Iterations: {}", iterations);

    // Create LoRA layer
    let config = LoRAConfig {
        rank,
        alpha,
        dropout: 0.0,
    };
    let lora = LoRALayer::new(in_features, out_features, &config, &device)
        .expect("Failed to create LoRA layer");

    // Create input tensor
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, in_features), &device).unwrap();

    println!("\n📊 Benchmarking LoRA Layer Forward Pass...");
    println!("   (Automatically uses fused kernel on Metal)");

    // Warmup
    for _ in 0..warmup {
        let _output = lora.forward(&input).unwrap();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _output = lora.forward(&input).unwrap();
    }
    let duration = start.elapsed();
    let avg_us = duration.as_micros() as f64 / iterations as f64;

    println!("\n  Total time: {:?}", duration);
    println!("  Average per iteration: {:.2} µs", avg_us);

    println!("\n🎯 Results:");
    println!("═══════════════════════════════════════════════════════");
    println!("  LoRA Forward: {:.2} µs", avg_us);

    // Compare to expected ranges
    println!("\n  Comparison:");
    println!("    MLX baseline:  5-11 µs");
    println!("    Candle only:   37-98 µs (unfused)");
    println!("    Our result:    {:.2} µs", avg_us);

    if avg_us < 15.0 {
        println!("\n✅ EXCELLENT! Using fused kernel successfully!");
    } else if avg_us < 30.0 {
        println!("\n⚠️  GOOD: Performance improved, but may not be using fused kernel");
    } else {
        println!("\n❌ SLOW: Likely falling back to unfused Candle operations");
    }

    println!();
}
