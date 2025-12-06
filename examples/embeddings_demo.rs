//! Demonstration of sentence-transformer embeddings for semantic search.
//!
//! This example shows how to:
//! 1. Load an embedding model from HuggingFace
//! 2. Generate embeddings for text
//! 3. Compute semantic similarity
//! 4. Perform simple semantic search
//!
//! Run with:
//! ```bash
//! cargo run --example embeddings_demo --features embeddings
//! ```

#![cfg(feature = "embeddings")]

use anyhow::Result;
use candle_core::Device;
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};

/// Compute cosine similarity between two normalized vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<()> {
    println!("🚀 Metal-Candle Embeddings Demo\n");

    // 1. Initialize device (CPU for portability, Metal GPU available on Apple Silicon)
    let device = Device::Cpu;
    println!("📱 Device: {:?}\n", device);

    // 2. Load embedding model (downloads from HuggingFace on first run)
    println!("📥 Loading E5-small-v2 model...");
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;
    println!("✅ Model loaded! Dimension: {}\n", model.dimension());

    // 3. Define a knowledge base
    let documents = vec![
        "Rust is a systems programming language focused on safety and performance.",
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons in the brain.",
        "The Transformer architecture revolutionized natural language processing.",
        "Metal is Apple's GPU acceleration framework for graphics and compute.",
    ];

    println!("📚 Knowledge base ({} documents):", documents.len());
    for (i, doc) in documents.iter().enumerate() {
        println!("  {}. {}", i + 1, doc);
    }
    println!();

    // 4. Generate embeddings for all documents
    println!("🔄 Generating embeddings...");
    let doc_embeddings = model.encode(&documents)?;
    println!("✅ Generated embeddings: {:?}\n", doc_embeddings.dims());

    // 5. Perform semantic search
    let queries = vec![
        "What is Rust programming?",
        "Tell me about deep learning",
        "GPU acceleration on Apple devices",
    ];

    println!("🔍 Semantic Search Results:\n");

    for query in &queries {
        println!("Query: \"{}\"", query);

        // Encode query
        let query_embedding = model.encode(&[*query])?;

        // Compute similarities
        let query_vec = query_embedding.to_vec2::<f32>()?;
        let doc_vecs = doc_embeddings.to_vec2::<f32>()?;

        let mut similarities: Vec<(usize, f32)> = doc_vecs
            .iter()
            .enumerate()
            .map(|(idx, doc_vec)| (idx, cosine_similarity(&query_vec[0], doc_vec)))
            .collect();

        // Sort by similarity (highest first)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Show top 3 results
        println!("  Top 3 matches:");
        for (rank, (idx, score)) in similarities.iter().take(3).enumerate() {
            println!(
                "    {}. [Score: {:.4}] {}",
                rank + 1,
                score,
                documents[*idx]
            );
        }
        println!();
    }

    // 6. Demonstrate similarity computation
    println!("🔗 Similarity Matrix:\n");
    println!("Computing pairwise similarities between first 4 documents...\n");

    let sample_docs = &documents[..4];
    let sample_embeddings = model.encode(sample_docs)?;
    let sample_vecs = sample_embeddings.to_vec2::<f32>()?;

    println!("     Doc1  Doc2  Doc3  Doc4");
    for (i, vec_i) in sample_vecs.iter().enumerate() {
        print!("Doc{} ", i + 1);
        for vec_j in &sample_vecs {
            let sim = cosine_similarity(vec_i, vec_j);
            print!(" {:.3}", sim);
        }
        println!();
    }
    println!();

    // 7. Show embedding statistics
    println!("📊 Embedding Statistics:\n");

    let all_vecs = doc_embeddings.to_vec2::<f32>()?;
    for (i, vec) in all_vecs.iter().take(3).enumerate() {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mean: f32 = vec.iter().sum::<f32>() / vec.len() as f32;
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);

        println!(
            "  Document {}: norm={:.6}, mean={:.6}, min={:.6}, max={:.6}",
            i + 1,
            norm,
            mean,
            min,
            max
        );
    }
    println!();

    // 8. Demonstrate batch processing
    println!("⚡ Batch Processing Demo:\n");

    let batch_sizes = [1, 2, 4, 8];
    for &batch_size in &batch_sizes {
        let batch: Vec<&str> = (0..batch_size)
            .map(|i| documents[i % documents.len()])
            .collect();

        let start = std::time::Instant::now();
        let _batch_embeddings = model.encode(&batch)?;
        let duration = start.elapsed();

        println!(
            "  Batch size {}: {:.2}ms ({:.2}ms per item)",
            batch_size,
            duration.as_secs_f64() * 1000.0,
            duration.as_secs_f64() * 1000.0 / batch_size as f64
        );
    }
    println!();

    println!("✨ Demo complete!");
    println!("\n💡 Tips:");
    println!("  - Models are cached in ~/.cache/ferris/models/");
    println!("  - Normalized embeddings: cosine similarity = dot product");
    println!("  - Use Metal GPU for faster encoding on Apple Silicon");

    Ok(())
}
