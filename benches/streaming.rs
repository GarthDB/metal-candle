//! Streaming inference benchmarks for metal-candle.
//!
//! Measures streaming performance including:
//! - Baseline non-streaming generation
//! - Sync streaming with callbacks
//! - Async streaming with futures
//! - Callback overhead
//! - Memory efficiency
//!
//! Run with: `cargo bench --bench streaming`

#![allow(missing_docs)]

use candle_core::{Device, Tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use metal_candle::error::Result as MetalResult;
use metal_candle::inference::{Generator, GeneratorConfig, SamplingStrategy};
use metal_candle::models::LanguageModel;

#[cfg(feature = "streaming")]
use futures::StreamExt;

/// Mock language model for benchmarking
struct MockModel {
    device: Device,
    vocab_size: usize,
}

impl MockModel {
    fn new(device: Device) -> Self {
        Self {
            device,
            vocab_size: 32000, // Typical vocabulary size
        }
    }
}

impl LanguageModel for MockModel {
    fn forward(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> MetalResult<Tensor> {
        let seq_len = input_ids.dims()[1];

        // Generate mock logits that simulate realistic token distributions
        // This is fast but representative of real model outputs
        let logits: Vec<f32> = (0..seq_len)
            .flat_map(|_| {
                (0..self.vocab_size)
                    .map(|i| {
                        // Create a distribution with some variance
                        let base = 5.0;
                        let decay = 0.0001;
                        base - (i as f32 * decay)
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        Ok(Tensor::from_vec(
            logits,
            (1, seq_len, self.vocab_size),
            &self.device,
        )?)
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

fn benchmark_baseline_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_generation");

    // Use CPU for consistent benchmarking
    let device = Device::Cpu;
    let input_ids = vec![1u32, 2, 3];

    for max_tokens in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("generate", max_tokens),
            &max_tokens,
            |b, &max_tokens| {
                b.iter(|| {
                    // Create generator for each iteration to avoid state issues
                    let model = MockModel::new(device.clone());
                    let config = GeneratorConfig {
                        max_tokens,
                        sampling: SamplingStrategy::Greedy,
                        temperature: 1.0,
                        repetition_penalty: 1.0,
                        stop_on_eos: false,
                        eos_token_id: None,
                        ..Default::default()
                    };

                    let mut generator = Generator::new(Box::new(model), config)
                        .expect("Failed to create generator");

                    let result = generator
                        .generate(black_box(&input_ids))
                        .expect("Generation failed");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_sync_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("sync_streaming");

    let device = Device::Cpu;
    let input_ids = vec![1u32, 2, 3];

    for max_tokens in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("generate_stream", max_tokens),
            &max_tokens,
            |b, &max_tokens| {
                b.iter(|| {
                    // Create generator for each iteration
                    let model = MockModel::new(device.clone());
                    let config = GeneratorConfig {
                        max_tokens,
                        sampling: SamplingStrategy::Greedy,
                        temperature: 1.0,
                        repetition_penalty: 1.0,
                        stop_on_eos: false,
                        eos_token_id: None,
                        ..Default::default()
                    };

                    let mut generator = Generator::new(Box::new(model), config)
                        .expect("Failed to create generator");

                    let result = generator
                        .generate_stream(black_box(&input_ids), |_token| {
                            // Minimal callback - just continue
                            true
                        })
                        .expect("Streaming failed");
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "streaming")]
fn benchmark_async_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_streaming");

    let device = Device::Cpu;
    let input_ids = vec![1u32, 2, 3];
    let runtime = tokio::runtime::Runtime::new().unwrap();

    for max_tokens in [10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("generate_stream_async", max_tokens),
            &max_tokens,
            |b, &max_tokens| {
                b.iter(|| {
                    runtime.block_on(async {
                        // Create generator for each iteration
                        let model = MockModel::new(device.clone());
                        let config = GeneratorConfig {
                            max_tokens,
                            sampling: SamplingStrategy::Greedy,
                            temperature: 1.0,
                            repetition_penalty: 1.0,
                            stop_on_eos: false,
                            eos_token_id: None,
                            ..Default::default()
                        };

                        let mut generator = Generator::new(Box::new(model), config)
                            .expect("Failed to create generator");

                        let stream = generator.generate_stream_async(black_box(&input_ids));

                        // Pin the stream to enable calling .next()
                        futures::pin_mut!(stream);

                        let mut tokens = Vec::new();
                        while let Some(result) = stream.next().await {
                            if let Ok(token) = result {
                                tokens.push(token.token_id);
                            }
                        }
                        black_box(tokens)
                    })
                });
            },
        );
    }

    group.finish();
}

fn benchmark_callback_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("callback_overhead");

    let device = Device::Cpu;
    let max_tokens = 50;
    let input_ids = vec![1u32, 2, 3];

    // Minimal callback
    group.bench_function("minimal_callback", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::Greedy,
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let result = generator
                .generate_stream(black_box(&input_ids), |_token| true)
                .expect("Streaming failed");
            black_box(result)
        });
    });

    // Callback with string formatting (typical use case)
    group.bench_function("formatting_callback", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::Greedy,
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let result = generator
                .generate_stream(black_box(&input_ids), |token| {
                    // Typical callback: format string (but don't print)
                    let _ = format!(
                        "Token {}: {:.2}%",
                        token.token_id,
                        token.probability * 100.0
                    );
                    true
                })
                .expect("Streaming failed");
            black_box(result)
        });
    });

    // Callback with accumulation (realistic use case)
    group.bench_function("accumulation_callback", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::Greedy,
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let mut accumulated = String::new();
            let result = generator
                .generate_stream(black_box(&input_ids), |token| {
                    // Realistic callback: accumulate text
                    if let Some(ref text) = token.text {
                        accumulated.push_str(text);
                    }
                    true
                })
                .expect("Streaming failed");
            black_box((result, accumulated))
        });
    });

    group.finish();
}

fn benchmark_sampling_strategies_streaming(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_strategies_streaming");

    let device = Device::Cpu;
    let max_tokens = 50;
    let input_ids = vec![1u32, 2, 3];

    // Greedy
    group.bench_function("greedy", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::Greedy,
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let result = generator
                .generate_stream(black_box(&input_ids), |_token| true)
                .expect("Streaming failed");
            black_box(result)
        });
    });

    // Top-k
    group.bench_function("top_k_50", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::TopK { k: 50 },
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let result = generator
                .generate_stream(black_box(&input_ids), |_token| true)
                .expect("Streaming failed");
            black_box(result)
        });
    });

    // Top-p
    group.bench_function("top_p_0.9", |b| {
        b.iter(|| {
            let model = MockModel::new(device.clone());
            let config = GeneratorConfig {
                max_tokens,
                sampling: SamplingStrategy::TopP { p: 0.9 },
                temperature: 1.0,
                repetition_penalty: 1.0,
                stop_on_eos: false,
                eos_token_id: None,
                ..Default::default()
            };

            let mut generator =
                Generator::new(Box::new(model), config).expect("Failed to create generator");

            let result = generator
                .generate_stream(black_box(&input_ids), |_token| true)
                .expect("Streaming failed");
            black_box(result)
        });
    });

    group.finish();
}

#[cfg(feature = "streaming")]
criterion_group!(
    streaming_benches,
    benchmark_baseline_generation,
    benchmark_sync_streaming,
    benchmark_async_streaming,
    benchmark_callback_overhead,
    benchmark_sampling_strategies_streaming,
);

#[cfg(not(feature = "streaming"))]
criterion_group!(
    streaming_benches,
    benchmark_baseline_generation,
    benchmark_sync_streaming,
    benchmark_callback_overhead,
    benchmark_sampling_strategies_streaming,
);

criterion_main!(streaming_benches);
