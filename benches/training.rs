//! Training performance benchmarks
//!
//! These benchmarks should be run locally on Apple Silicon hardware.
//! Results will vary based on hardware and should not be run in CI.

#![allow(missing_docs)]

use criterion::{criterion_group, criterion_main, Criterion};

fn bench_placeholder(_c: &mut Criterion) {
    // Placeholder - will be implemented in Phase 3
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
