//! Benchmarks comparing lazy evaluation vs eager execution.
//!
//! This benchmark suite measures the overhead of lazy evaluation and async execution
//! to provide data-driven insights for optimization decisions.

#![allow(missing_docs)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use metal_candle::graph::LazyTensor;
use metal_candle::training::{LoRAConfig, LoRALayer};

use candle_core::{Device as CandleDevice, Tensor};

/// Benchmark basic tensor operations: lazy vs eager
fn benchmark_basic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_operations");

    let device = CandleDevice::Cpu;
    let data_a = vec![1.0f32; 1024];
    let data_b = vec![2.0f32; 1024];

    // Eager: Direct Candle operations
    group.bench_function("eager_add", |b| {
        b.iter(|| {
            let a = Tensor::from_slice(&data_a, &[1024], &device).unwrap();
            let b = Tensor::from_slice(&data_b, &[1024], &device).unwrap();
            let c = (&a + &b).unwrap();
            black_box(c);
        });
    });

    // Lazy: Graph building + evaluation
    group.bench_function("lazy_add_sync", |b| {
        b.iter(|| {
            let a_tensor = Tensor::from_slice(&data_a, &[1024], &device).unwrap();
            let b_tensor = Tensor::from_slice(&data_b, &[1024], &device).unwrap();

            let a = LazyTensor::from_tensor(a_tensor).unwrap();
            let b = a.add_tensor_to_graph(b_tensor).unwrap();
            let c = a.add(&b).unwrap();
            let result = c.eval().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark matrix multiplication: lazy vs eager
fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");

    let device = CandleDevice::Cpu;

    for size in [64, 128, 256] {
        let data_a = vec![1.0f32; size * size];
        let data_b = vec![2.0f32; size * size];

        group.throughput(Throughput::Elements((size * size * size) as u64));

        // Eager matmul
        group.bench_with_input(BenchmarkId::new("eager", size), &size, |b, &_size| {
            b.iter(|| {
                let a = Tensor::from_slice(&data_a, &[size, size], &device).unwrap();
                let b = Tensor::from_slice(&data_b, &[size, size], &device).unwrap();
                let c = a.matmul(&b).unwrap();
                black_box(c);
            });
        });

        // Lazy matmul
        group.bench_with_input(BenchmarkId::new("lazy_sync", size), &size, |b, &_size| {
            b.iter(|| {
                let a_tensor = Tensor::from_slice(&data_a, &[size, size], &device).unwrap();
                let b_tensor = Tensor::from_slice(&data_b, &[size, size], &device).unwrap();

                let a = LazyTensor::from_tensor(a_tensor).unwrap();
                let b = a.add_tensor_to_graph(b_tensor).unwrap();
                let c = a.matmul(&b).unwrap();
                let result = c.eval().unwrap();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark LoRA operations: lazy vs eager
fn benchmark_lora(c: &mut Criterion) {
    let mut group = c.benchmark_group("lora");

    let device = CandleDevice::Cpu;

    let config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
    };

    let lora = LoRALayer::new(512, 512, &config, &device).unwrap();
    let input_data = vec![1.0f32; 4 * 512];

    // Eager LoRA (using forward)
    group.bench_function("eager_lora", |b| {
        b.iter(|| {
            let input = Tensor::from_slice(&input_data, &[4, 512], &device).unwrap();
            let output = lora.forward(&input).unwrap();
            black_box(output);
        });
    });

    // Lazy LoRA (using forward_lazy)
    #[cfg(feature = "graph")]
    group.bench_function("lazy_lora_sync", |b| {
        b.iter(|| {
            let input = Tensor::from_slice(&input_data, &[4, 512], &device).unwrap();
            let input_lazy = LazyTensor::from_tensor(input).unwrap();
            let output = lora.forward_lazy(&input_lazy).unwrap();
            let result = output.eval().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark complex computation graphs
fn benchmark_complex_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_graph");

    let device = CandleDevice::Cpu;

    let a_data = vec![1.0f32; 64 * 64];
    let b_data = vec![1.0f32; 64 * 64];
    let c_data = vec![0.5f32; 64 * 64];

    // Eager: Sequential operations
    group.bench_function("eager_chain", |b| {
        b.iter(|| {
            let a = Tensor::from_slice(&a_data, &[64, 64], &device).unwrap();
            let b = Tensor::from_slice(&b_data, &[64, 64], &device).unwrap();
            let c = Tensor::from_slice(&c_data, &[64, 64], &device).unwrap();

            // a @ b + c * 2.0 -> softmax
            let matmul = a.matmul(&b).unwrap();
            let add = (&matmul + &c).unwrap();
            let scaled = add.affine(2.0, 0.0).unwrap();
            let result = candle_nn::ops::softmax(&scaled, 1).unwrap();
            black_box(result);
        });
    });

    // Lazy: Build graph then execute
    group.bench_function("lazy_chain_sync", |b| {
        b.iter(|| {
            let a =
                LazyTensor::from_tensor(Tensor::from_slice(&a_data, &[64, 64], &device).unwrap())
                    .unwrap();
            let b = a
                .add_tensor_to_graph(Tensor::from_slice(&b_data, &[64, 64], &device).unwrap())
                .unwrap();
            let c = a
                .add_tensor_to_graph(Tensor::from_slice(&c_data, &[64, 64], &device).unwrap())
                .unwrap();

            // Build graph
            let matmul = a.matmul(&b).unwrap();
            let add = matmul.add(&c).unwrap();
            let scaled = add.mul_scalar(2.0).unwrap();
            let soft = scaled.softmax(1).unwrap();

            // Execute
            let result = soft.eval().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark graph building overhead
fn benchmark_graph_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_building");

    let device = CandleDevice::Cpu;
    let data = vec![1.0f32; 1024];

    // Just graph building (no execution)
    group.bench_function("build_graph_no_eval", |b| {
        b.iter(|| {
            let a = LazyTensor::from_tensor(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();
            let b = a
                .add_tensor_to_graph(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();

            // Build 10 operations
            let mut current = a.add(&b).unwrap();
            for _ in 0..9 {
                current = current.mul_scalar(1.1).unwrap();
            }

            black_box(current);
        });
    });

    // Graph building + execution
    group.bench_function("build_and_eval", |b| {
        b.iter(|| {
            let a = LazyTensor::from_tensor(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();
            let b = a
                .add_tensor_to_graph(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();

            // Build 10 operations
            let mut current = a.add(&b).unwrap();
            for _ in 0..9 {
                current = current.mul_scalar(1.1).unwrap();
            }

            // Execute
            let result = current.eval().unwrap();
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark async execution overhead (if async-exec feature enabled)
#[cfg(feature = "async-exec")]
fn benchmark_async_overhead(c: &mut Criterion) {
    use tokio::runtime::Runtime;

    let mut group = c.benchmark_group("async_overhead");

    let rt = Runtime::new().unwrap();
    let device = CandleDevice::Cpu;
    let data = vec![1.0f32; 1024];

    // Sync evaluation
    group.bench_function("sync_eval", |b| {
        b.iter(|| {
            let a = LazyTensor::from_tensor(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();
            let b = a
                .add_tensor_to_graph(Tensor::from_slice(&data, &[1024], &device).unwrap())
                .unwrap();
            let c = a.add(&b).unwrap();
            let result = c.eval().unwrap();
            black_box(result);
        });
    });

    // Async evaluation
    group.bench_function("async_eval", |b| {
        b.iter(|| {
            rt.block_on(async {
                let a =
                    LazyTensor::from_tensor(Tensor::from_slice(&data, &[1024], &device).unwrap())
                        .unwrap();
                let b = a
                    .add_tensor_to_graph(Tensor::from_slice(&data, &[1024], &device).unwrap())
                    .unwrap();
                let c = a.add(&b).unwrap();
                let result = c.eval_async().await.unwrap();
                black_box(result);
            });
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_basic_operations,
    benchmark_matmul,
    benchmark_lora,
    benchmark_complex_graph,
    benchmark_graph_building,
);

#[cfg(feature = "async-exec")]
criterion_group!(async_benches, benchmark_async_overhead);

#[cfg(feature = "async-exec")]
criterion_main!(benches, async_benches);

#[cfg(not(feature = "async-exec"))]
criterion_main!(benches);
