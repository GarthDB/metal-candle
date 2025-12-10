//! Benchmark embeddings with different batch sizes to find GPU crossover point.

#![allow(warnings)]

#[cfg(feature = "embeddings")]
use candle_core::Device;
#[cfg(feature = "embeddings")]
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

fn main() {
    #[cfg(not(feature = "embeddings"))]
    {
        eprintln!("This benchmark requires the 'embeddings' feature.");
        std::process::exit(1);
    }

    #[cfg(feature = "embeddings")]
    run_benchmark();
}

#[cfg(feature = "embeddings")]
fn run_benchmark() {
    println!("📊 Embeddings Batch Size Benchmark\n");
    println!("Finding the CPU vs Metal crossover point...\n");

    // Create sample text (typical RAG document length)
    let sample_text = "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It accomplishes these goals without a garbage collector, making it a useful language for embedded systems and other performance-critical applications.";

    // Test different batch sizes
    let batch_sizes = vec![1, 2, 5, 10, 20, 50, 100];

    println!("Batch Size | CPU Time | Metal Time | Speedup   | Winner");
    println!("-----------|----------|------------|-----------|--------");

    for &batch_size in &batch_sizes {
        // Create batch
        let texts: Vec<&str> = (0..batch_size).map(|_| sample_text).collect();

        // CPU benchmark
        let cpu_device = Device::Cpu;
        let cpu_model =
            match EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, cpu_device) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Failed to load CPU model: {}", e);
                    return;
                }
            };

        let cpu_start = std::time::Instant::now();
        for _ in 0..3 {
            let _ = cpu_model.encode(&texts).unwrap();
        }
        let cpu_time = cpu_start.elapsed().as_micros() / 3;

        // Metal benchmark
        let (metal_time, speedup, winner) = if let Ok(metal_device) = Device::new_metal(0) {
            match EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, metal_device) {
                Ok(metal_model) => {
                    let metal_start = std::time::Instant::now();
                    for _ in 0..3 {
                        let _ = metal_model.encode(&texts).unwrap();
                    }
                    let mt = metal_start.elapsed().as_micros() / 3;
                    let speedup = cpu_time as f64 / mt as f64;
                    let winner = if speedup > 1.0 { "Metal 🚀" } else { "CPU" };
                    (mt, speedup, winner)
                }
                Err(_) => (0, 0.0, "N/A"),
            }
        } else {
            (0, 0.0, "N/A")
        };

        if metal_time > 0 {
            println!(
                "{:10} | {:7}µs | {:9}µs | {:8.2}x | {}",
                batch_size, cpu_time, metal_time, speedup, winner
            );
        } else {
            println!(
                "{:10} | {:7}µs | Metal N/A  | N/A       | CPU",
                batch_size, cpu_time
            );
        }
    }

    println!("\n📈 Analysis:");
    println!("  • Crossover point: Where speedup > 1.0x");
    println!("  • For Ferris RAG: Use batch size ≥ crossover for best performance");
    println!("  • Single queries: CPU is fine (low latency matters)");
    println!("  • Bulk indexing: Use largest feasible batch size");
}
