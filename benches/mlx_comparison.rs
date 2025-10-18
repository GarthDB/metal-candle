//! MLX vs metal-candle Performance Comparison
//!
//! This benchmark compares metal-candle performance against MLX+PyO3 baseline.
//! It runs the Python MLX baseline script and compares results with Rust benchmarks.
//!
//! Run with: `cargo bench --bench mlx_comparison`

#![allow(missing_docs)]

use candle_core::{Device, Tensor};
use metal_candle::backend::TensorExt;
use metal_candle::training::{LoRAConfig, LoRALayer};
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

fn run_mlx_baseline() -> Result<HashMap<String, HashMap<String, f64>>, Box<dyn std::error::Error>> {
    println!("\n🐍 Running MLX baseline benchmarks...");
    println!("════════════════════════════════════════════════\n");

    let output = Command::new("python3")
        .arg("benches/mlx_baseline.py")
        .output()?;

    if !output.status.success() {
        eprintln!("MLX baseline failed:");
        eprintln!("{}", String::from_utf8_lossy(&output.stderr));
        return Err("MLX baseline script failed".into());
    }

    // Print MLX output
    println!("{}", String::from_utf8_lossy(&output.stdout));

    // Parse results from JSON file
    let json_content = std::fs::read_to_string("mlx_baseline_results.json")?;
    let results: HashMap<String, HashMap<String, f64>> = serde_json::from_str(&json_content)?;

    Ok(results)
}

fn benchmark_metal_candle_lora() -> HashMap<String, f64> {
    println!("\n🦀 Running metal-candle benchmarks...");
    println!("════════════════════════════════════════════════\n");

    let mut results = HashMap::new();
    let device = Device::new_metal(0).expect("Metal required");
    let iterations = 100;

    let configs = vec![
        ("small_512x512_r8", 512, 512, 8),
        ("medium_1024x1024_r8", 1024, 1024, 8),
        ("large_2048x2048_r8", 2048, 2048, 8),
    ];

    println!("LoRA Forward:");
    for (name, in_features, out_features, rank) in configs {
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let layer = LoRALayer::new(in_features, out_features, &config, &device)
            .expect("Failed to create layer");

        let input = Tensor::randn(0f32, 1f32, (1, 1, in_features), &device)
            .expect("Failed to create input");

        // Warmup
        for _ in 0..10 {
            let _ = layer.forward(&input);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = layer.forward(&input).expect("Forward failed");
        }
        let elapsed = start.elapsed();
        let avg_us = elapsed.as_micros() as f64 / iterations as f64;

        println!("  {}: {:.2} µs", name, avg_us);
        results.insert(name.to_string(), avg_us);
    }

    results
}

fn benchmark_metal_candle_layer_ops() -> HashMap<String, f64> {
    let mut results = HashMap::new();
    let device = Device::new_metal(0).expect("Metal required");
    let iterations = 100;
    let size = 1024;

    let tensor =
        Tensor::randn(0f32, 1f32, (4, 16, size), &device).expect("Failed to create tensor");

    println!("\nLayer Operations:");

    // Softmax
    let _ = tensor.softmax_stable(); // Warmup
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tensor.softmax_stable().expect("Softmax failed");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  softmax_stable: {:.2} µs", avg_us);
    results.insert("softmax_stable".to_string(), avg_us);

    // Layer Norm
    let _ = tensor.layer_norm(1e-5); // Warmup
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tensor.layer_norm(1e-5).expect("LayerNorm failed");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  layer_norm: {:.2} µs", avg_us);
    results.insert("layer_norm".to_string(), avg_us);

    // RMS Norm
    let _ = tensor.rms_norm(1e-5); // Warmup
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tensor.rms_norm(1e-5).expect("RMSNorm failed");
    }
    let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;
    println!("  rms_norm: {:.2} µs", avg_us);
    results.insert("rms_norm".to_string(), avg_us);

    results
}

fn benchmark_metal_candle_rank_scaling() -> HashMap<String, f64> {
    let mut results = HashMap::new();
    let device = Device::new_metal(0).expect("Metal required");
    let iterations = 100;
    let in_features = 1024;
    let out_features = 1024;

    println!("\nLoRA Rank Scaling:");
    for rank in [4, 8, 16, 32, 64] {
        #[allow(clippy::cast_precision_loss)]
        let alpha = (rank * 2) as f32;
        let config = LoRAConfig {
            rank,
            alpha,
            dropout: 0.0,
        };

        let layer = LoRALayer::new(in_features, out_features, &config, &device)
            .expect("Failed to create layer");

        let input = Tensor::randn(0f32, 1f32, (1, 1, in_features), &device)
            .expect("Failed to create input");

        // Warmup
        for _ in 0..10 {
            let _ = layer.forward(&input);
        }

        // Benchmark
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = layer.forward(&input).expect("Forward failed");
        }
        let avg_us = start.elapsed().as_micros() as f64 / iterations as f64;

        println!("  rank_{}: {:.2} µs", rank, avg_us);
        results.insert(format!("rank_{}", rank), avg_us);
    }

    results
}

fn compare_and_report(
    mlx_results: &HashMap<String, HashMap<String, f64>>,
    rust_lora: &HashMap<String, f64>,
    rust_ops: &HashMap<String, f64>,
    rust_scaling: &HashMap<String, f64>,
) {
    println!("\n");
    println!("═══════════════════════════════════════════════════════════════");
    println!("                    COMPARISON RESULTS");
    println!("═══════════════════════════════════════════════════════════════");

    // LoRA Forward comparison
    if let Some(mlx_lora) = mlx_results.get("lora_forward") {
        println!("\n📊 LoRA Forward Pass:");
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "{:<30} {:>12} {:>12} {:>10}",
            "Operation", "MLX (µs)", "Rust (µs)", "Speedup"
        );
        println!("─────────────────────────────────────────────────────────────");

        for (name, &rust_time) in rust_lora {
            if let Some(&mlx_time) = mlx_lora.get(name) {
                let speedup = mlx_time / rust_time;
                let symbol = if speedup >= 1.0 { "🚀" } else { "⚠️ " };
                println!(
                    "{:<30} {:>12.2} {:>12.2} {:>9.2}x {}",
                    name, mlx_time, rust_time, speedup, symbol
                );
            }
        }
    }

    // Layer operations comparison
    if let Some(mlx_ops) = mlx_results.get("layer_operations") {
        println!("\n📊 Layer Operations:");
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "{:<30} {:>12} {:>12} {:>10}",
            "Operation", "MLX (µs)", "Rust (µs)", "Speedup"
        );
        println!("─────────────────────────────────────────────────────────────");

        for (name, &rust_time) in rust_ops {
            if let Some(&mlx_time) = mlx_ops.get(name) {
                let speedup = mlx_time / rust_time;
                let symbol = if speedup >= 1.0 { "🚀" } else { "⚠️ " };
                println!(
                    "{:<30} {:>12.2} {:>12.2} {:>9.2}x {}",
                    name, mlx_time, rust_time, speedup, symbol
                );
            }
        }
    }

    // Rank scaling comparison
    if let Some(mlx_scaling) = mlx_results.get("lora_rank_scaling") {
        println!("\n📊 LoRA Rank Scaling:");
        println!("─────────────────────────────────────────────────────────────");
        println!(
            "{:<30} {:>12} {:>12} {:>10}",
            "Rank", "MLX (µs)", "Rust (µs)", "Speedup"
        );
        println!("─────────────────────────────────────────────────────────────");

        for (name, &rust_time) in rust_scaling {
            if let Some(&mlx_time) = mlx_scaling.get(name) {
                let speedup = mlx_time / rust_time;
                let symbol = if speedup >= 1.0 { "🚀" } else { "⚠️ " };
                println!(
                    "{:<30} {:>12.2} {:>12.2} {:>9.2}x {}",
                    name, mlx_time, rust_time, speedup, symbol
                );
            }
        }
    }

    // Overall summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                         SUMMARY");
    println!("═══════════════════════════════════════════════════════════════");

    let mut total_comparisons = 0;
    let mut faster_count = 0;
    let mut total_speedup = 0.0;

    // Calculate statistics
    if let Some(mlx_lora) = mlx_results.get("lora_forward") {
        for (name, &rust_time) in rust_lora {
            if let Some(&mlx_time) = mlx_lora.get(name) {
                total_comparisons += 1;
                let speedup = mlx_time / rust_time;
                total_speedup += speedup;
                if speedup >= 1.0 {
                    faster_count += 1;
                }
            }
        }
    }

    if let Some(mlx_ops) = mlx_results.get("layer_operations") {
        for (name, &rust_time) in rust_ops {
            if let Some(&mlx_time) = mlx_ops.get(name) {
                total_comparisons += 1;
                let speedup = mlx_time / rust_time;
                total_speedup += speedup;
                if speedup >= 1.0 {
                    faster_count += 1;
                }
            }
        }
    }

    if total_comparisons > 0 {
        let avg_speedup = total_speedup / total_comparisons as f64;
        let faster_percent = (faster_count as f64 / total_comparisons as f64) * 100.0;

        println!("\nTotal operations compared: {}", total_comparisons);
        println!(
            "metal-candle faster: {} ({:.1}%)",
            faster_count, faster_percent
        );
        println!("Average speedup: {:.2}x", avg_speedup);

        let target_met = avg_speedup >= 0.9; // 90% of MLX performance
        let target_symbol = if target_met { "✅" } else { "❌" };
        println!(
            "\nPerformance target (90-100% of MLX): {} {}",
            if target_met { "MET" } else { "NOT MET" },
            target_symbol
        );

        if avg_speedup >= 1.0 {
            println!(
                "\n🎉 metal-candle is FASTER than MLX by {:.1}%!",
                (avg_speedup - 1.0) * 100.0
            );
        } else {
            println!(
                "\n⚠️  metal-candle is {:.1}% slower than MLX",
                (1.0 - avg_speedup) * 100.0
            );
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════\n");
}

fn main() {
    println!("\n🔬 MLX vs metal-candle Performance Comparison");
    println!("════════════════════════════════════════════════");

    // Run MLX baseline
    let mlx_results = match run_mlx_baseline() {
        Ok(results) => results,
        Err(e) => {
            eprintln!("Error running MLX baseline: {}", e);
            eprintln!("\nMake sure MLX is installed: pip install mlx");
            std::process::exit(1);
        }
    };

    // Run metal-candle benchmarks
    let rust_lora = benchmark_metal_candle_lora();
    let rust_ops = benchmark_metal_candle_layer_ops();
    let rust_scaling = benchmark_metal_candle_rank_scaling();

    // Compare and report
    compare_and_report(&mlx_results, &rust_lora, &rust_ops, &rust_scaling);
}
