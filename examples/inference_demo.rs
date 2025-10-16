//! Inference demonstration: KV-cache and sampling strategies.
//!
//! This example demonstrates:
//! 1. KV-cache usage for efficient generation
//! 2. Multiple sampling strategies
//! 3. Performance comparison
//!
//! Run with: `cargo run --example inference_demo`

use anyhow::Result;
use candle_core::{Device, Tensor};
use metal_candle::inference::{sample_token, KVCache, KVCacheConfig, SamplingStrategy};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 Metal-Candle Inference Demo");
    println!("════════════════════════════════════════════════");
    println!();

    // Setup
    let device = Device::Cpu;

    // Demo 1: KV-Cache
    demo_kv_cache(&device)?;

    // Demo 2: Sampling Strategies
    demo_sampling(&device)?;

    // Demo 3: Performance Comparison
    demo_performance(&device)?;

    println!("✅ Demo complete!");

    Ok(())
}

fn demo_kv_cache(device: &Device) -> Result<()> {
    println!("📦 Demo 1: KV-Cache for Efficient Generation");
    println!("────────────────────────────────────────────────");

    let config = KVCacheConfig {
        max_seq_len: 128,
        num_layers: 4,
        num_heads: 8,
        head_dim: 64,
        batch_size: 1,
    };

    let mut cache = KVCache::new(config.clone(), device)?;

    println!("Configuration:");
    println!("  Max sequence length: {}", config.max_seq_len);
    println!("  Number of layers: {}", config.num_layers);
    println!("  Number of heads: {}", config.num_heads);
    println!("  Dimension per head: {}", config.head_dim);
    println!();

    // Simulate adding tokens to cache
    println!("Simulating token generation with cache:");

    for step in 1..=5 {
        let key = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_heads, 1, config.head_dim),
            device,
        )?;
        let value = Tensor::randn(
            0f32,
            1f32,
            (1, config.num_heads, 1, config.head_dim),
            device,
        )?;

        let start = Instant::now();
        let (full_key, _full_value) = cache.update(0, &key, &value)?;
        let elapsed = start.elapsed();

        println!(
            "  Step {}: Cached {} tokens | Key shape: {:?} | Time: {:?}",
            step,
            cache.position(),
            full_key.dims(),
            elapsed
        );
    }

    println!();
    println!("Cache statistics:");
    println!("  Total tokens cached: {}", cache.position());
    println!("  Layers in cache: {}", cache.num_cached_layers());
    println!("  Cache full: {}", cache.position() >= cache.max_seq_len());
    println!();

    // Memory estimation
    let total_elements = config.num_layers * 2 // key + value
        * config.batch_size
        * config.num_heads
        * cache.position()
        * config.head_dim;
    let memory_bytes = total_elements * 4; // f32 = 4 bytes
    #[allow(clippy::cast_precision_loss)] // Memory size estimation, precision loss acceptable
    let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;

    println!("Memory usage (estimated): {memory_mb:.2} MB");
    println!();

    Ok(())
}

fn demo_sampling(device: &Device) -> Result<()> {
    println!("🎲 Demo 2: Sampling Strategies");
    println!("────────────────────────────────────────────────");

    // Create mock logits (10 tokens, skewed distribution)
    let logits_data: Vec<f32> = vec![
        1.0, // token 0
        5.0, // token 1 (most likely)
        3.0, // token 2
        0.5, // token 3
        2.0, // token 4
        0.1, // token 5
        4.0, // token 6
        1.5, // token 7
        0.8, // token 8
        2.5, // token 9
    ];

    let logits = Tensor::new(logits_data, device)?;

    println!("Logits distribution:");
    println!("  Token 1: 5.0 (highest)");
    println!("  Token 6: 4.0");
    println!("  Token 2: 3.0");
    println!("  Token 9: 2.5");
    println!("  ... (others lower)");
    println!();

    // Test each sampling strategy
    println!("Sampling Results:");
    println!();

    // 1. Greedy
    println!("1. Greedy Sampling (deterministic):");
    let strategy = SamplingStrategy::Greedy;
    let token = sample_token(&logits, &strategy)?;
    println!("   Selected token: {token} (always picks highest logit)");
    println!();

    // 2. Top-k
    println!("2. Top-k Sampling (k=3):");
    let strategy = SamplingStrategy::TopK { k: 3 };
    println!("   Candidates: top 3 tokens [1, 6, 2]");
    for _ in 0..5 {
        let token = sample_token(&logits, &strategy)?;
        print!("   Sample: {token} ");
    }
    println!();
    println!();

    // 3. Top-p
    println!("3. Top-p Sampling (nucleus, p=0.9):");
    let strategy = SamplingStrategy::TopP { p: 0.9 };
    println!("   Adaptive: includes tokens until cumulative prob ≥ 0.9");
    for _ in 0..5 {
        let token = sample_token(&logits, &strategy)?;
        print!("   Sample: {token} ");
    }
    println!();
    println!();

    // 4. Temperature
    println!("4. Temperature Sampling:");
    for &temp in &[0.5, 1.0, 2.0] {
        let strategy = SamplingStrategy::Temperature { temperature: temp };
        print!("   T={temp}: ");
        for _ in 0..5 {
            let token = sample_token(&logits, &strategy)?;
            print!("{token} ");
        }
        println!();
    }
    println!("   (Lower T = more deterministic, Higher T = more random)");
    println!();

    Ok(())
}

fn demo_performance(device: &Device) -> Result<()> {
    println!("⚡ Demo 3: Performance Comparison");
    println!("────────────────────────────────────────────────");

    let vocab_size = 32000;
    let num_samples = 100;

    println!("Benchmarking {num_samples} samples with vocab size {vocab_size}");
    println!();

    // Create random logits
    let logits = Tensor::randn(0f32, 1f32, vocab_size, device)?;

    // Benchmark each strategy
    let strategies = vec![
        ("Greedy", SamplingStrategy::Greedy),
        ("Top-k (k=50)", SamplingStrategy::TopK { k: 50 }),
        ("Top-p (p=0.9)", SamplingStrategy::TopP { p: 0.9 }),
        (
            "Temperature (T=0.7)",
            SamplingStrategy::Temperature { temperature: 0.7 },
        ),
    ];

    for (name, strategy) in strategies {
        let start = Instant::now();

        for _ in 0..num_samples {
            let _ = sample_token(&logits, &strategy)?;
        }

        let elapsed = start.elapsed();
        #[allow(clippy::cast_precision_loss)] // Microseconds for timing, precision loss acceptable
        let avg_micros = elapsed.as_micros() as f64 / f64::from(num_samples);

        println!("  {name:<25} Avg: {avg_micros:>8.2} μs/sample");
    }

    println!();
    println!("Note: Sampling overhead is minimal (<1% of typical model forward pass)");
    println!();

    Ok(())
}
