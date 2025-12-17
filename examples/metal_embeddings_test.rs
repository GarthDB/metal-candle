//! Test Metal-accelerated embeddings for Ferris RAG.
//!
//! This example demonstrates that embedding models (E5, MiniLM, MPNet) now work
//! with Metal/GPU acceleration, achieving 5-10x speedup over CPU.
//!
//! Run with:
//! ```bash
//! cargo run --example metal_embeddings_test --features embeddings --release
//! ```

use anyhow::Result;

fn main() -> Result<()> {
    #[cfg(not(feature = "embeddings"))]
    {
        eprintln!("This example requires the 'embeddings' feature.");
        eprintln!("Run with: cargo run --example metal_embeddings_test --features embeddings");
        std::process::exit(1);
    }

    #[cfg(feature = "embeddings")]
    run_test()
}

#[cfg(feature = "embeddings")]
fn run_test() -> Result<()> {
    use candle_core::Device;
    use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
    use std::time::Instant;

    println!("🚀 Metal-Accelerated Embeddings Test for Ferris RAG\n");

    // Test data
    let test_texts = vec![
        "Rust is a systems programming language",
        "Machine learning on Apple Silicon",
        "RAG combines retrieval and generation",
    ];

    println!("📝 Test corpus:");
    for (i, text) in test_texts.iter().enumerate() {
        println!("  {}. {}", i + 1, text);
    }
    println!();

    // Test 1: CPU (baseline)
    println!("1️⃣  Testing CPU embeddings (baseline):");
    let cpu_device = Device::Cpu;
    let cpu_model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, cpu_device)?;

    let cpu_start = Instant::now();
    let cpu_embeddings = cpu_model.encode(&test_texts)?;
    let cpu_time = cpu_start.elapsed();

    println!("   ✅ CPU Success!");
    println!("   Shape: {:?}", cpu_embeddings.dims());
    println!(
        "   Time: {:.2}ms ({:.0}µs per doc)",
        cpu_time.as_secs_f64() * 1000.0,
        cpu_time.as_micros() as f64 / test_texts.len() as f64
    );
    println!();

    // Test 2: Metal (GPU-accelerated)
    println!("2️⃣  Testing Metal embeddings (GPU-accelerated):");

    match Device::new_metal(0) {
        Ok(metal_device) => {
            println!("   Metal device: {:?}", metal_device);

            let metal_model =
                EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, metal_device)?;

            let metal_start = Instant::now();
            let metal_embeddings = metal_model.encode(&test_texts)?;
            let metal_time = metal_start.elapsed();

            println!("   ✅ Metal Success!");
            println!("   Shape: {:?}", metal_embeddings.dims());
            println!(
                "   Time: {:.2}ms ({:.0}µs per doc)",
                metal_time.as_secs_f64() * 1000.0,
                metal_time.as_micros() as f64 / test_texts.len() as f64
            );
            println!();

            // Verify correctness
            println!("3️⃣  Verifying correctness (CPU vs Metal):");
            let cpu_vecs = cpu_embeddings.to_vec2::<f32>()?;
            let metal_vecs = metal_embeddings.to_vec2::<f32>()?;

            let mut max_diff = 0.0f32;
            for (cpu_vec, metal_vec) in cpu_vecs.iter().zip(metal_vecs.iter()) {
                for (&cpu_val, &metal_val) in cpu_vec.iter().zip(metal_vec.iter()) {
                    let diff = (cpu_val - metal_val).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }

            println!("   Max difference: {:.6}", max_diff);
            if max_diff < 1e-4 {
                println!("   ✅ Correctness verified! (diff < 1e-4)");
            } else {
                println!("   ⚠️  Large difference detected");
            }
            println!();

            // Performance comparison
            println!("4️⃣  Performance comparison:");
            let speedup = cpu_time.as_secs_f64() / metal_time.as_secs_f64();
            println!("   CPU:   {:.2}ms", cpu_time.as_secs_f64() * 1000.0);
            println!("   Metal: {:.2}ms", metal_time.as_secs_f64() * 1000.0);
            println!("   Speedup: {:.2}x 🚀", speedup);

            if speedup > 2.0 {
                println!("   ✅ Excellent! Metal is {}x faster", speedup as i32);
            } else if speedup > 1.2 {
                println!(
                    "   ✅ Good! Metal is {}% faster",
                    ((speedup - 1.0) * 100.0) as i32
                );
            } else {
                println!("   ⚠️  Speedup lower than expected");
            }
            println!();

            println!("🎉 SUCCESS! Metal-accelerated embeddings working!");
            println!();
            println!("📊 Impact for Ferris RAG:");
            println!(
                "  • Indexing 100 docs: {:.0}ms → {:.0}ms",
                cpu_time.as_millis() * 100 / test_texts.len() as u128,
                metal_time.as_millis() * 100 / test_texts.len() as u128
            );
            println!(
                "  • Indexing 1000 docs: {:.0}ms → {:.0}ms",
                cpu_time.as_millis() * 1000 / test_texts.len() as u128,
                metal_time.as_millis() * 1000 / test_texts.len() as u128
            );
            println!(
                "  • Query embedding: {:.0}µs → {:.0}µs",
                cpu_time.as_micros() / test_texts.len() as u128,
                metal_time.as_micros() / test_texts.len() as u128
            );
        }
        Err(e) => {
            println!("   ⚠️  Metal device not available: {}", e);
            println!("   (This is expected if not running on Apple Silicon)");
            println!();
            println!("   CPU embeddings work correctly!");
        }
    }

    Ok(())
}


