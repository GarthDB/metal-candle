//! Integration tests for inference components.

use candle_core::{DType, Device, Tensor};
use metal_candle::inference::{
    apply_repetition_penalty, sample_token, Generator, GeneratorConfig, KVCache, KVCacheConfig,
    SamplingStrategy,
};
use metal_candle::models::LanguageModel;

#[test]
fn test_kv_cache_multi_layer_integration() {
    let device = Device::Cpu;
    let config = KVCacheConfig {
        max_seq_len: 10,
        num_layers: 3,
        num_heads: 4,
        head_dim: 8,
        batch_size: 1,
    };

    let mut cache = KVCache::new(config, &device).unwrap();

    // Simulate 3 layers processing 2 tokens
    for layer_idx in 0..3 {
        // Token 1
        let key1 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value1 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        cache.update(layer_idx, &key1, &value1).unwrap();

        // Token 2
        let key2 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let value2 = Tensor::ones((1, 4, 1, 8), DType::F32, &device).unwrap();
        let (full_key, full_value) = cache.update(layer_idx, &key2, &value2).unwrap();

        // Verify accumulated sequence length
        assert_eq!(full_key.dims(), &[1, 4, 2, 8]);
        assert_eq!(full_value.dims(), &[1, 4, 2, 8]);
    }

    assert_eq!(cache.position(), 2);
    assert_eq!(cache.num_cached_layers(), 3);
}

#[test]
fn test_kv_cache_reset_and_reuse() {
    let device = Device::Cpu;
    let config = KVCacheConfig {
        max_seq_len: 10,
        num_layers: 2,
        num_heads: 2,
        head_dim: 4,
        batch_size: 1,
    };

    let mut cache = KVCache::new(config, &device).unwrap();

    // First sequence
    let key = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
    let value = Tensor::zeros((1, 2, 1, 4), DType::F32, &device).unwrap();
    cache.update(0, &key, &value).unwrap();
    cache.update(0, &key, &value).unwrap();

    assert_eq!(cache.position(), 2);

    // Clear and start new sequence
    cache.clear();
    assert_eq!(cache.position(), 0);
    assert!(cache.is_empty());

    // Second sequence
    cache.update(0, &key, &value).unwrap();
    assert_eq!(cache.position(), 1);
}

#[test]
fn test_sampling_consistency() {
    let device = Device::Cpu;

    // Create deterministic logits
    let logits = Tensor::new(&[1.0f32, 5.0, 3.0, 2.0], &device).unwrap();

    // Greedy should always return the same token
    let strategy = SamplingStrategy::Greedy;
    let token1 = sample_token(&logits, &strategy, &[], 1.0).unwrap();
    let token2 = sample_token(&logits, &strategy, &[], 1.0).unwrap();
    let token3 = sample_token(&logits, &strategy, &[], 1.0).unwrap();

    assert_eq!(token1, token2);
    assert_eq!(token2, token3);
    assert_eq!(token1, 1); // Index of max value (5.0)
}

#[test]
fn test_repetition_penalty_integration() {
    let device = Device::Cpu;
    let mut logits = Tensor::new(&[1.0f32, 5.0, 3.0, 2.0], &device).unwrap();

    // Apply penalty to token 1 (highest logit)
    apply_repetition_penalty(&mut logits, &[1], 2.0).unwrap();

    let result = logits.to_vec1::<f32>().unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5); // 5.0 / 2.0 = 2.5
    assert!((result[2] - 3.0).abs() < 1e-5); // Unchanged
}

#[test]
fn test_sampling_top_k_bounds() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[1.0f32, 5.0, 3.0, 2.0], &device).unwrap();

    // Top-k should only sample from top k tokens
    let strategy = SamplingStrategy::TopK { k: 2 };

    // Sample multiple times and verify all samples are from top-2 [1, 2]
    for _ in 0..20 {
        let token = sample_token(&logits, &strategy, &[], 1.0).unwrap();
        assert!(token == 1 || token == 2, "Token {token} not in top-2");
    }
}

#[test]
fn test_sampling_strategies_valid_output() {
    let device = Device::Cpu;
    let vocab_size = 100;
    let logits = Tensor::randn(0f32, 1f32, vocab_size, &device).unwrap();

    let strategies = vec![
        SamplingStrategy::Greedy,
        SamplingStrategy::TopK { k: 10 },
        SamplingStrategy::TopP { p: 0.9 },
        SamplingStrategy::Temperature { temperature: 0.8 },
    ];

    for strategy in strategies {
        let token = sample_token(&logits, &strategy, &[], 1.0).unwrap();
        let vocab_size_u32 = u32::try_from(vocab_size).unwrap_or(u32::MAX);
        assert!(token < vocab_size_u32, "Token ID exceeds vocab size");
    }
}

#[test]
fn test_generator_config_defaults() {
    let config = GeneratorConfig::default();

    assert_eq!(config.max_tokens, 100);
    assert!((config.temperature - 1.0).abs() < 1e-7);
    assert!(config.eos_token_id.is_none());
    assert!(matches!(config.sampling, SamplingStrategy::Greedy));
}

// Mock model for testing
struct MockLanguageModel {
    device: Device,
    vocab_size: usize,
}

impl MockLanguageModel {
    fn new(vocab_size: usize) -> Self {
        Self {
            device: Device::Cpu,
            vocab_size,
        }
    }
}

impl LanguageModel for MockLanguageModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> metal_candle::error::Result<Tensor> {
        let seq_len = input_ids.dims()[1];
        // Return decreasing logits for simplicity
        let mut logits_vec = Vec::new();
        for _ in 0..seq_len {
            for i in 0..self.vocab_size {
                logits_vec.push((self.vocab_size - i) as f32);
            }
        }
        Ok(Tensor::from_vec(
            logits_vec,
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

#[test]
fn test_generator_creation_with_model() {
    let model = MockLanguageModel::new(100);
    let config = GeneratorConfig {
        max_tokens: 50,
        sampling: SamplingStrategy::TopP { p: 0.9 },
        temperature: 0.7,
        top_p: Some(0.9),
        repetition_penalty: 1.1,
        eos_token_id: Some(2),
        ..Default::default()
    };

    let generator = Generator::new(Box::new(model), config).unwrap();
    assert_eq!(generator.config().max_tokens, 50);
    assert!((generator.config().temperature - 0.7).abs() < 1e-7);
}

#[test]
fn test_generator_basic_generation_integration() {
    let model = MockLanguageModel::new(50);
    let config = GeneratorConfig {
        max_tokens: 10,
        sampling: SamplingStrategy::Greedy,
        ..Default::default()
    };

    let mut generator = Generator::new(Box::new(model), config).unwrap();
    let input_ids = vec![1u32, 2, 3];
    let output = generator.generate(&input_ids).unwrap();

    // Should generate input + max_tokens
    assert!(output.len() <= input_ids.len() + 10);
    // First tokens should match input
    assert_eq!(&output[..input_ids.len()], &input_ids[..]);
}

#[test]
fn test_generator_streaming_integration() {
    let model = MockLanguageModel::new(50);
    let config = GeneratorConfig {
        max_tokens: 5,
        sampling: SamplingStrategy::Greedy,
        ..Default::default()
    };

    let mut generator = Generator::new(Box::new(model), config).unwrap();
    let input_ids = vec![1u32];

    let mut streamed = Vec::new();
    let output = generator
        .generate_stream(&input_ids, |token| {
            streamed.push(token);
            true
        })
        .unwrap();

    // All generated tokens should be captured
    assert_eq!(streamed.len(), output.len() - input_ids.len());
}

#[test]
fn test_kv_cache_memory_efficiency() {
    let device = Device::Cpu;
    let config = KVCacheConfig {
        max_seq_len: 100,
        num_layers: 1,
        num_heads: 2,
        head_dim: 4,
        batch_size: 1,
    };

    let mut cache = KVCache::new(config.clone(), &device).unwrap();

    // Add 10 tokens
    for _ in 0..10 {
        let key = Tensor::ones((1, 2, 1, 4), DType::F32, &device).unwrap();
        let value = Tensor::ones((1, 2, 1, 4), DType::F32, &device).unwrap();
        cache.update(0, &key, &value).unwrap();
    }

    // Verify cache only stores what's needed
    let (key, value) = cache.get(0).unwrap();
    assert_eq!(key.dims(), &[1, 2, 10, 4]); // Only 10 tokens, not max_seq_len
    assert_eq!(value.dims(), &[1, 2, 10, 4]);
}

#[test]
fn test_kv_cache_concurrent_layers() {
    let device = Device::Cpu;
    let config = KVCacheConfig {
        max_seq_len: 20,
        num_layers: 5,
        num_heads: 4,
        head_dim: 8,
        batch_size: 1,
    };

    let mut cache = KVCache::new(config, &device).unwrap();

    // Process 3 tokens across all 5 layers
    for _ in 0..3 {
        for layer_idx in 0..5 {
            let key = Tensor::randn(0f32, 1f32, (1, 4, 1, 8), &device).unwrap();
            let value = Tensor::randn(0f32, 1f32, (1, 4, 1, 8), &device).unwrap();
            cache.update(layer_idx, &key, &value).unwrap();
        }
    }

    // Verify all layers cached
    assert_eq!(cache.num_cached_layers(), 5);
    assert_eq!(cache.position(), 3);

    // Verify each layer has the right sequence length
    for layer_idx in 0..5 {
        let (key, value) = cache.get(layer_idx).unwrap();
        assert_eq!(key.dims()[2], 3); // seq_len dimension
        assert_eq!(value.dims()[2], 3);
    }
}

#[test]
fn test_sampling_with_edge_cases() {
    let device = Device::Cpu;

    // Single token vocabulary
    let logits_single = Tensor::new(&[1.0f32], &device).unwrap();
    let token = sample_token(&logits_single, &SamplingStrategy::Greedy, &[], 1.0).unwrap();
    assert_eq!(token, 0);

    // All equal logits
    let logits_equal = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &device).unwrap();
    let token = sample_token(&logits_equal, &SamplingStrategy::Greedy, &[], 1.0).unwrap();
    assert!(token < 4);

    // Very skewed distribution
    let logits_skewed = Tensor::new(&[100.0f32, 0.0, 0.0, 0.0], &device).unwrap();
    let token = sample_token(
        &logits_skewed,
        &SamplingStrategy::TopP { p: 0.99 },
        &[],
        1.0,
    )
    .unwrap();
    assert_eq!(token, 0); // Should almost always pick the dominant token
}
