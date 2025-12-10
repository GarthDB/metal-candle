//! Integration tests for embeddings module.
//!
//! These tests download actual models from HuggingFace Hub and verify
//! end-to-end functionality. They are slower than unit tests but provide
//! comprehensive validation.

#![cfg(feature = "embeddings")]

use approx::assert_relative_eq;
use metal_candle::embeddings::{EmbeddingConfig, EmbeddingModel, EmbeddingModelType};
use metal_candle::Device;

/// Helper function to compute cosine similarity (dot product for normalized vectors)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
fn test_embedding_model_load() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    assert_eq!(model.dimension(), 384);
    assert_eq!(model.model_type(), EmbeddingModelType::E5SmallV2);

    Ok(())
}

#[test]
fn test_embedding_shape() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts = vec!["Hello world", "Test sentence"];
    let embeddings = model.encode(&texts)?;

    // Check shape: [batch=2, dim=384]
    assert_eq!(embeddings.dims(), &[2, 384]);

    Ok(())
}

#[test]
fn test_single_text_encoding() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts = vec!["Single sentence test"];
    let embeddings = model.encode(&texts)?;

    assert_eq!(embeddings.dims(), &[1, 384]);

    Ok(())
}

#[test]
fn test_batch_encoding() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    // Test various batch sizes
    for batch_size in [1, 2, 4, 8] {
        let texts: Vec<&str> = (0..batch_size).map(|i| {
            if i % 2 == 0 {
                "Even sentence"
            } else {
                "Odd sentence"
            }
        }).collect();

        let embeddings = model.encode(&texts)?;
        assert_eq!(embeddings.dims()[0], batch_size);
        assert_eq!(embeddings.dims()[1], 384);
    }

    Ok(())
}

#[test]
fn test_normalization() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let text = vec!["Test normalization"];
    let embeddings = model.encode(&text)?;

    // Check L2 norm is 1.0 (normalized)
    let vec = embeddings.to_vec2::<f32>()?;
    let norm: f32 = vec[0].iter().map(|x| x * x).sum::<f32>().sqrt();

    assert_relative_eq!(norm, 1.0, epsilon = 1e-5);

    Ok(())
}

#[test]
fn test_semantic_similarity() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts = vec![
        "The cat sits on the mat",
        "A cat is sitting on a mat",
        "Python is a programming language",
    ];

    let embeddings = model.encode(&texts)?;
    let vecs = embeddings.to_vec2::<f32>()?;

    // Cosine similarity (dot product for normalized vectors)
    let sim_cat_cat = cosine_similarity(&vecs[0], &vecs[1]);
    let sim_cat_python = cosine_similarity(&vecs[0], &vecs[2]);

    // Similar sentences should have higher similarity
    assert!(sim_cat_cat > sim_cat_python, "sim_cat_cat={}, sim_cat_python={}", sim_cat_cat, sim_cat_python);
    assert!(sim_cat_cat > 0.7, "Similar sentences should have high similarity: {}", sim_cat_cat);

    Ok(())
}

#[test]
fn test_empty_input_error() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts: Vec<&str> = vec![];
    let result = model.encode(&texts);

    assert!(result.is_err(), "Empty input should return an error");

    Ok(())
}

#[test]
fn test_long_text_truncation() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    // Create a very long text (exceeds max_seq_length=512)
    let long_text = vec!["word "].repeat(1000).join("");
    let texts = vec![long_text.as_str()];

    // Should handle truncation gracefully
    let embeddings = model.encode(&texts)?;
    assert_eq!(embeddings.dims(), &[1, 384]);

    Ok(())
}

#[test]
fn test_model_caching() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // First load (may download if not cached)
    let model1 = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device.clone())?;

    // Second load (should be fast, from cache)
    let model2 = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    // Both should produce identical embeddings
    let text = vec!["Cache test"];
    let emb1 = model1.encode(&text)?;
    let emb2 = model2.encode(&text)?;

    let vec1 = emb1.to_vec2::<f32>()?;
    let vec2 = emb2.to_vec2::<f32>()?;

    for (a, b) in vec1[0].iter().zip(&vec2[0]) {
        assert_relative_eq!(a, b, epsilon = 1e-5);
    }

    Ok(())
}

#[test]
fn test_custom_config() -> anyhow::Result<()> {
    let config = EmbeddingConfig {
        model_type: EmbeddingModelType::E5SmallV2,
        normalize: false, // Disable normalization
        max_seq_length: 256,
    };

    let device = Device::Cpu;
    let model = EmbeddingModel::from_config(config, device)?;

    let text = vec!["Test custom config"];
    let embeddings = model.encode(&text)?;

    // Check shape is correct
    assert_eq!(embeddings.dims(), &[1, 384]);

    // Norm might not be 1.0 since normalization is disabled
    let vec = embeddings.to_vec2::<f32>()?;
    let norm: f32 = vec[0].iter().map(|x| x * x).sum::<f32>().sqrt();

    // Norm should be non-zero but might not be 1.0
    assert!(norm > 0.0);

    Ok(())
}

#[test]
fn test_different_model_types() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // Test E5SmallV2 (384 dim)
    let model_e5 = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device.clone())?;
    assert_eq!(model_e5.dimension(), 384);

    let text = vec!["Test sentence"];
    let embeddings = model_e5.encode(&text)?;
    assert_eq!(embeddings.dims(), &[1, 384]);

    // Note: Testing other models (MiniLM, MPNet) would require downloading them
    // We keep it to E5SmallV2 for faster CI

    Ok(())
}

#[test]
fn test_special_characters() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts = vec![
        "Hello, world!",
        "Test with émojis 🚀 and spécial çharacters",
        "Code: fn main() { println!(\"Hello\"); }",
        "Math: x² + y² = z²",
    ];

    let embeddings = model.encode(&texts)?;
    assert_eq!(embeddings.dims(), &[4, 384]);

    Ok(())
}

#[test]
fn test_multilingual_text() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let texts = vec![
        "Hello",
        "Bonjour",
        "Hola",
        "こんにちは",
    ];

    // Should handle different languages
    let embeddings = model.encode(&texts)?;
    assert_eq!(embeddings.dims(), &[4, 384]);

    // All embeddings should be normalized
    let vecs = embeddings.to_vec2::<f32>()?;
    for vec in vecs {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
    }

    Ok(())
}

#[test]
fn test_deterministic_embeddings() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model = EmbeddingModel::from_pretrained(EmbeddingModelType::E5SmallV2, device)?;

    let text = vec!["Determinism test"];

    // Encode same text multiple times
    let emb1 = model.encode(&text)?;
    let emb2 = model.encode(&text)?;
    let emb3 = model.encode(&text)?;

    let vec1 = emb1.to_vec2::<f32>()?;
    let vec2 = emb2.to_vec2::<f32>()?;
    let vec3 = emb3.to_vec2::<f32>()?;

    // Should produce identical results
    for i in 0..384 {
        assert_relative_eq!(vec1[0][i], vec2[0][i], epsilon = 1e-6);
        assert_relative_eq!(vec1[0][i], vec3[0][i], epsilon = 1e-6);
    }

    Ok(())
}

// Note: Metal GPU tests would go here if we want to test device-specific behavior
// For now, we focus on CPU tests for broad compatibility


