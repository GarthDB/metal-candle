//! Inference benchmarks for metal-candle.
//!
//! Measures text generation and inference performance including:
//! - KV-cache operations
//! - Token sampling strategies
//! - Forward pass latency
//! - Memory efficiency
//!
//! Run with: `cargo bench --bench inference`

#![allow(missing_docs)]

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use metal_candle::inference::{sample_token, KVCache, KVCacheConfig, SamplingStrategy};

fn benchmark_kv_cache_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_update");

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    // Qwen 0.5B config
    let config = KVCacheConfig {
        max_seq_len: 2048,
        num_layers: 24,
        num_heads: 14,
        head_dim: 64,
        batch_size: 1,
    };

    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    let mut cache = KVCache::new(config, &device).expect("Failed to create cache");

    let key = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create key");

    let value = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create value");

    group.bench_function("single_layer_update", |b| {
        b.iter(|| {
            cache.clear(); // Clear before each iteration
            let result = cache.update(0, &key, &value).expect("Cache update failed");
            black_box(result)
        });
    });

    group.bench_function("all_layers_update", |b| {
        b.iter(|| {
            cache.clear(); // Clear before each iteration
            for layer_idx in 0..num_layers {
                let result = cache
                    .update(layer_idx, &key, &value)
                    .expect("Cache update failed");
                black_box(result);
            }
        });
    });

    group.finish();
}

fn benchmark_kv_cache_retrieval(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_retrieval");

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    let config = KVCacheConfig {
        max_seq_len: 2048,
        num_layers: 24,
        num_heads: 14,
        head_dim: 64,
        batch_size: 1,
    };

    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    let mut cache = KVCache::new(config, &device).expect("Failed to create cache");

    // Populate cache with some data
    let key = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create key");

    let value = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create value");

    // Add 100 tokens to cache
    for _ in 0..100 {
        for layer_idx in 0..num_layers {
            cache
                .update(layer_idx, &key, &value)
                .expect("Cache update failed");
        }
    }

    group.bench_function("retrieve_from_cache", |b| {
        b.iter(|| {
            // Just clear and update - measures cache operation cost
            cache.clear();
            let result = cache.update(0, &key, &value).expect("Cache access failed");
            black_box(result)
        });
    });

    group.finish();
}

fn benchmark_sampling_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_strategies");

    let device = Device::new_metal(0).expect("Metal required for benchmarks");
    let vocab_size = 32000;

    // Create logits (typical model output)
    let logits = Tensor::randn(0f32, 1f32, vocab_size, &device).expect("Failed to create logits");

    group.throughput(Throughput::Elements(vocab_size as u64));

    // Greedy sampling
    let strategy_greedy = SamplingStrategy::Greedy;
    group.bench_function("greedy", |b| {
        b.iter(|| {
            let token = sample_token(black_box(&logits), &strategy_greedy, &[], 1.0)
                .expect("Greedy sampling failed");
            black_box(token)
        });
    });

    // Top-k sampling
    for k in [10, 40, 50] {
        let strategy = SamplingStrategy::TopK { k };
        group.bench_with_input(
            BenchmarkId::new("top_k", format!("k_{k}")),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    let token = sample_token(black_box(&logits), strategy, &[], 1.0)
                        .expect("Top-k sampling failed");
                    black_box(token)
                });
            },
        );
    }

    // Top-p sampling
    for (p, label) in [(0.9, "90"), (0.95, "95")] {
        let strategy = SamplingStrategy::TopP { p };
        group.bench_with_input(
            BenchmarkId::new("top_p", format!("p_{label}")),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    let token = sample_token(black_box(&logits), strategy, &[], 1.0)
                        .expect("Top-p sampling failed");
                    black_box(token)
                });
            },
        );
    }

    // Temperature sampling
    for (temp, label) in [(0.7, "7"), (0.8, "8"), (1.0, "10")] {
        let strategy = SamplingStrategy::Temperature { temperature: temp };
        group.bench_with_input(
            BenchmarkId::new("temperature", format!("t_{label}")),
            &strategy,
            |b, strategy| {
                b.iter(|| {
                    let token = sample_token(black_box(&logits), strategy, &[], 1.0)
                        .expect("Temperature sampling failed");
                    black_box(token)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_kv_cache_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_scaling");
    group.sample_size(10); // Fewer samples for memory-intensive operations

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    // Test different sequence lengths
    for seq_len in [512, 1024, 2048] {
        let config = KVCacheConfig {
            max_seq_len: seq_len,
            num_layers: 24,
            num_heads: 14,
            head_dim: 64,
            batch_size: 1,
        };

        let num_heads = config.num_heads;
        let head_dim = config.head_dim;
        let num_layers_local = config.num_layers;

        let key = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
            .expect("Failed to create key");

        let value = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
            .expect("Failed to create value");

        group.bench_function(format!("full_sequence_seq_{seq_len}"), |b| {
            b.iter(|| {
                let config_local = KVCacheConfig {
                    max_seq_len: seq_len,
                    num_layers: num_layers_local,
                    num_heads,
                    head_dim,
                    batch_size: 1,
                };
                let mut local_cache =
                    KVCache::new(config_local, &device).expect("Failed to create cache");

                // Fill cache for half the sequence length
                for _ in 0..(seq_len / 2) {
                    for layer_idx in 0..num_layers_local {
                        local_cache
                            .update(layer_idx, &key, &value)
                            .expect("Cache update failed");
                    }
                }
                black_box(local_cache)
            });
        });
    }

    group.finish();
}

fn benchmark_sampling_vocab_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_vocab_scaling");

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    // Test different vocabulary sizes
    for vocab_size in [1000, 10_000, 32_000, 100_000] {
        let logits =
            Tensor::randn(0f32, 1f32, vocab_size, &device).expect("Failed to create logits");

        group.throughput(Throughput::Elements(vocab_size as u64));

        // Greedy (should be O(n) for argmax)
        let strategy_greedy = SamplingStrategy::Greedy;
        group.bench_with_input(
            BenchmarkId::new("greedy", format!("vocab_{vocab_size}")),
            &(&logits, &strategy_greedy),
            |b, (logits, strategy)| {
                b.iter(|| {
                    let token = sample_token(black_box(logits), strategy, &[], 1.0)
                        .expect("Greedy sampling failed");
                    black_box(token)
                });
            },
        );

        // Top-k (should be O(n + k log k))
        let strategy_topk = SamplingStrategy::TopK { k: 50 };
        group.bench_with_input(
            BenchmarkId::new("top_k_50", format!("vocab_{vocab_size}")),
            &(&logits, &strategy_topk),
            |b, (logits, strategy)| {
                b.iter(|| {
                    let token = sample_token(black_box(logits), strategy, &[], 1.0)
                        .expect("Top-k sampling failed");
                    black_box(token)
                });
            },
        );

        // Top-p (requires sorting, O(n log n))
        let strategy_top_p = SamplingStrategy::TopP { p: 0.9 };
        group.bench_with_input(
            BenchmarkId::new("top_p_0.9", format!("vocab_{vocab_size}")),
            &(&logits, &strategy_top_p),
            |b, (logits, strategy)| {
                b.iter(|| {
                    let token = sample_token(black_box(logits), strategy, &[], 1.0)
                        .expect("Top-p sampling failed");
                    black_box(token)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_kv_cache_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("kv_cache_memory");
    group.sample_size(10);

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    // Different model sizes
    let configs = vec![
        ("qwen_0.5B", 24, 14, 64), // Qwen 0.5B
        ("qwen_1.5B", 28, 16, 64), // Qwen 1.5B (hypothetical)
        ("qwen_3B", 32, 32, 64),   // Qwen 3B (hypothetical)
    ];

    for (name, num_layers, num_heads, head_dim) in configs {
        group.bench_function(format!("allocate_{name}"), |b| {
            b.iter(|| {
                let config_local = KVCacheConfig {
                    max_seq_len: 2048,
                    num_layers,
                    num_heads,
                    head_dim,
                    batch_size: 1,
                };
                let cache = KVCache::new(config_local, &device).expect("Failed to create cache");
                black_box(cache)
            });
        });
    }

    group.finish();
}

fn benchmark_generation_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation_simulation");
    group.sample_size(10);

    let device = Device::new_metal(0).expect("Metal required for benchmarks");

    let config = KVCacheConfig {
        max_seq_len: 2048,
        num_layers: 24,
        num_heads: 14,
        head_dim: 64,
        batch_size: 1,
    };

    let num_layers = config.num_layers;
    let num_heads = config.num_heads;
    let head_dim = config.head_dim;

    let mut cache = KVCache::new(config, &device).expect("Failed to create cache");

    let key = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create key");

    let value = Tensor::randn(0f32, 1f32, (1, num_heads, 1, head_dim), &device)
        .expect("Failed to create value");

    let logits = Tensor::randn(0f32, 1f32, 32000, &device).expect("Failed to create logits");

    group.bench_function("token_generation_cycle", |b| {
        b.iter(|| {
            cache.clear(); // Clear before each iteration

            // Simulate one token generation:
            // 1. Update KV-cache for all layers
            for layer_idx in 0..num_layers {
                cache
                    .update(layer_idx, &key, &value)
                    .expect("Cache update failed");
            }

            // 2. Sample next token
            let strategy = SamplingStrategy::TopP { p: 0.9 };
            let token = sample_token(&logits, &strategy, &[], 1.0).expect("Sampling failed");

            black_box(token)
        });
    });

    group.finish();
}

criterion_group!(
    inference_benches,
    benchmark_kv_cache_update,
    benchmark_kv_cache_retrieval,
    benchmark_sampling_strategies,
    benchmark_kv_cache_scaling,
    benchmark_sampling_vocab_size,
    benchmark_kv_cache_memory,
    benchmark_generation_simulation,
);
criterion_main!(inference_benches);
