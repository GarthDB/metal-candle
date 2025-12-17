//! Simple manual benchmark for fused LoRA kernel.

use candle_core::{Device, Tensor};
use metal_candle::backend::custom_ops::FusedLoRAOp;
use std::time::Instant;

fn main() {
    println!("\n🚀 Fused LoRA Kernel Performance Test");
    println!("════════════════════════════════════════");

    let device = Device::new_metal(0).expect("Metal device required");

    // Test configuration
    let batch_size = 1;
    let seq_len = 128;
    let features = 512;
    let rank = 8;
    let scaling = 2.0f32;
    let iterations = 100;
    let warmup = 10;

    println!("\nConfiguration:");
    println!("  Batch size: {}", batch_size);
    println!("  Sequence length: {}", seq_len);
    println!("  Features: {}", features);
    println!("  Rank: {}", rank);
    println!("  Iterations: {}", iterations);

    // Create test tensors
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_len, features), &device).unwrap();
    let lora_a = Tensor::randn(0f32, 0.01f32, (features, rank), &device).unwrap();
    let lora_b = Tensor::randn(0f32, 0.01f32, (rank, features), &device).unwrap();

    // Benchmark unfused (standard Candle)
    println!("\n📊 Benchmarking UNFUSED (Candle)...");

    // Warmup
    for _ in 0..warmup {
        let hidden = input.broadcast_matmul(&lora_a).unwrap();
        let output = hidden.broadcast_matmul(&lora_b).unwrap();
        let _scaled = output.affine(f64::from(scaling), 0.0).unwrap();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let hidden = input.broadcast_matmul(&lora_a).unwrap();
        let output = hidden.broadcast_matmul(&lora_b).unwrap();
        let _scaled = output.affine(f64::from(scaling), 0.0).unwrap();
    }
    let unfused_duration = start.elapsed();
    let unfused_avg_us = unfused_duration.as_micros() as f64 / iterations as f64;

    println!("  Total time: {:?}", unfused_duration);
    println!("  Average per iteration: {:.2} µs", unfused_avg_us);

    // Benchmark fused (custom Metal kernel)
    println!("\n📊 Benchmarking FUSED (Metal kernel)...");

    // Warmup
    for _ in 0..warmup {
        let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), scaling).unwrap();
        let _output = input.apply_op1(op).unwrap();
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let op = FusedLoRAOp::new(lora_a.clone(), lora_b.clone(), scaling).unwrap();
        let _output = input.apply_op1(op).unwrap();
    }
    let fused_duration = start.elapsed();
    let fused_avg_us = fused_duration.as_micros() as f64 / iterations as f64;

    println!("  Total time: {:?}", fused_duration);
    println!("  Average per iteration: {:.2} µs", fused_avg_us);

    // Calculate speedup
    let speedup = unfused_avg_us / fused_avg_us;

    println!("\n🎯 Results:");
    println!("════════════════════════════════════════");
    println!("  Unfused (Candle):  {:.2} µs", unfused_avg_us);
    println!("  Fused (Metal):     {:.2} µs", fused_avg_us);
    println!("  Speedup:           {:.2}x", speedup);

    if speedup >= 6.0 {
        println!(
            "\n✅ SUCCESS! Achieved {:.2}x speedup (target: 6-10x)",
            speedup
        );
    } else if speedup >= 3.0 {
        println!(
            "\n⚠️  PARTIAL: Achieved {:.2}x speedup (target: 6-10x)",
            speedup
        );
    } else {
        println!(
            "\n❌ BELOW TARGET: Achieved {:.2}x speedup (target: 6-10x)",
            speedup
        );
    }

    println!();
}


