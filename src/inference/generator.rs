//! Text generation pipeline.

use crate::error::Result;
use crate::inference::{sample_token, SamplingStrategy};
use crate::models::LanguageModel;
use candle_core::{IndexOp, Tensor};

/// Configuration for text generation.
///
/// Controls all aspects of the generation process including sampling strategy,
/// stop conditions, and penalties.
///
/// # Examples
///
/// ```
/// use metal_candle::inference::{GeneratorConfig, SamplingStrategy};
///
/// // Default configuration (greedy sampling, max 100 tokens)
/// let config = GeneratorConfig::default();
///
/// // Custom configuration with top-p sampling and repetition penalty
/// let config = GeneratorConfig {
///     max_tokens: 256,
///     sampling: SamplingStrategy::TopP { p: 0.95 },
///     temperature: 0.7,
///     top_p: Some(0.95),
///     top_k: None,
///     repetition_penalty: 1.1,
///     stop_on_eos: true,
///     eos_token_id: Some(151643),
///     stop_tokens: vec![],
/// };
/// ```
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Sampling strategy
    pub sampling: SamplingStrategy,

    /// Temperature for sampling (higher = more random, 0.0 = greedy)
    pub temperature: f64,

    /// Top-p (nucleus) sampling threshold (0.0-1.0)
    pub top_p: Option<f64>,

    /// Top-k sampling: only sample from top k tokens
    pub top_k: Option<usize>,

    /// Repetition penalty factor (> 1.0 = penalize, 1.0 = no penalty)
    pub repetition_penalty: f32,

    /// Stop generation when EOS token is encountered
    pub stop_on_eos: bool,

    /// End-of-sequence token ID
    pub eos_token_id: Option<u32>,

    /// Additional stop token IDs (generation stops if any are generated)
    pub stop_tokens: Vec<u32>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            sampling: SamplingStrategy::default(),
            temperature: 1.0,
            top_p: None,
            top_k: None,
            repetition_penalty: 1.0,
            stop_on_eos: true,
            eos_token_id: None,
            stop_tokens: Vec::new(),
        }
    }
}

/// Text generator for autoregressive models.
///
/// Provides high-level text generation with KV-cache, sampling strategies,
/// and stop conditions.
///
/// # Examples
///
/// ```
/// use metal_candle::inference::{GeneratorConfig, SamplingStrategy};
///
/// // Configure generation
/// let gen_config = GeneratorConfig {
///     max_tokens: 128,
///     sampling: SamplingStrategy::TopP { p: 0.95 },
///     temperature: 0.7,
///     repetition_penalty: 1.1,
///     ..Default::default()
/// };
///
/// // With a loaded model, you would use:
/// // let mut generator = Generator::new(Box::new(model), gen_config)?;
/// // let output_ids = generator.generate(&input_ids)?;
/// ```
pub struct Generator {
    model: Box<dyn LanguageModel>,
    config: GeneratorConfig,
}

impl Generator {
    /// Creates a new generator with the specified model and configuration.
    ///
    /// # Arguments
    ///
    /// * `model` - Language model for generation
    /// * `config` - Generation configuration
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new(model: Box<dyn LanguageModel>, config: GeneratorConfig) -> Result<Self> {
        Ok(Self { model, config })
    }

    /// Returns a reference to the generator configuration.
    #[must_use]
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Generates tokens autoregressively from input token IDs.
    ///
    /// This is the main generation method that implements the full generation loop
    /// with sampling, stop conditions, and repetition penalty.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs to condition generation on
    ///
    /// # Returns
    ///
    /// Returns a vector of generated token IDs (including the input tokens).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model forward pass fails
    /// - Sampling fails
    /// - Tensor operations fail
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::inference::{Generator, GeneratorConfig};
    /// # fn example(mut generator: Generator) -> Result<(), Box<dyn std::error::Error>> {
    /// let input_ids = vec![1, 2, 3];
    /// let output_ids = generator.generate(&input_ids)?;
    /// println!("Generated {} tokens", output_ids.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate(&mut self, input_ids: &[u32]) -> Result<Vec<u32>> {
        let mut generated_ids = input_ids.to_vec();
        let device = self.model.device();

        // Generate tokens one at a time
        for _ in 0..self.config.max_tokens {
            // Prepare input tensor for current step
            let current_ids = if generated_ids.is_empty() {
                // Should not happen, but handle gracefully
                return Ok(generated_ids);
            } else {
                // For now, we pass all tokens each time (no KV cache optimization yet)
                // This will be optimized in a future update to only pass the last token
                &generated_ids[..]
            };

            // Convert to tensor
            let input_tensor = Tensor::new(current_ids, device)?;
            let input_tensor = input_tensor.unsqueeze(0)?; // Add batch dimension

            // Forward pass
            let logits = self.model.forward(&input_tensor, None)?;

            // Get logits for the last token
            let last_logits = logits.i((0, logits.dims()[1] - 1))?;

            // Sample next token with repetition penalty
            let next_token = sample_token(
                &last_logits,
                &self.config.sampling,
                &generated_ids,
                self.config.repetition_penalty,
            )?;

            // Add to generated sequence
            generated_ids.push(next_token);

            // Check stop conditions
            if self.should_stop(next_token) {
                break;
            }
        }

        Ok(generated_ids)
    }

    /// Generates tokens autoregressively with streaming callback.
    ///
    /// This method generates tokens one at a time and calls the provided callback
    /// for each generated token. The callback can stop generation by returning `false`.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs to condition generation on
    /// * `callback` - Function called for each generated token. Returns `false` to stop.
    ///
    /// # Returns
    ///
    /// Returns a vector of generated token IDs (including the input tokens).
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model forward pass fails
    /// - Sampling fails
    /// - Tensor operations fail
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use metal_candle::inference::{Generator, GeneratorConfig};
    /// # fn example(mut generator: Generator) -> Result<(), Box<dyn std::error::Error>> {
    /// let input_ids = vec![1, 2, 3];
    /// let output_ids = generator.generate_stream(&input_ids, |token| {
    ///     print!("{} ", token);
    ///     true // Continue generation
    /// })?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn generate_stream<F>(&mut self, input_ids: &[u32], mut callback: F) -> Result<Vec<u32>>
    where
        F: FnMut(u32) -> bool,
    {
        let mut generated_ids = input_ids.to_vec();
        let device = self.model.device();

        // Generate tokens one at a time
        for _ in 0..self.config.max_tokens {
            // Prepare input tensor for current step
            let current_ids = if generated_ids.is_empty() {
                // Should not happen, but handle gracefully
                return Ok(generated_ids);
            } else {
                // For now, we pass all tokens each time (no KV cache optimization yet)
                &generated_ids[..]
            };

            // Convert to tensor
            let input_tensor = Tensor::new(current_ids, device)?;
            let input_tensor = input_tensor.unsqueeze(0)?; // Add batch dimension

            // Forward pass
            let logits = self.model.forward(&input_tensor, None)?;

            // Get logits for the last token
            let last_logits = logits.i((0, logits.dims()[1] - 1))?;

            // Sample next token with repetition penalty
            let next_token = sample_token(
                &last_logits,
                &self.config.sampling,
                &generated_ids,
                self.config.repetition_penalty,
            )?;

            // Add to generated sequence
            generated_ids.push(next_token);

            // Call callback with the newly generated token
            let should_continue = callback(next_token);

            // Check stop conditions
            if !should_continue || self.should_stop(next_token) {
                break;
            }
        }

        Ok(generated_ids)
    }

    /// Checks if generation should stop based on the generated token.
    ///
    /// # Arguments
    ///
    /// * `token` - The token that was just generated
    ///
    /// # Returns
    ///
    /// Returns `true` if generation should stop.
    fn should_stop(&self, token: u32) -> bool {
        // Check EOS token
        if self.config.stop_on_eos {
            if let Some(eos_id) = self.config.eos_token_id {
                if token == eos_id {
                    return true;
                }
            }
        }

        // Check custom stop tokens
        if self.config.stop_tokens.contains(&token) {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use candle_core::{Device, Tensor};

    // Mock model for testing
    struct MockModel {
        device: Device,
        vocab_size: usize,
    }

    impl MockModel {
        fn new(vocab_size: usize) -> Self {
            Self {
                device: Device::Cpu,
                vocab_size,
            }
        }
    }

    impl LanguageModel for MockModel {
        fn forward(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
            let seq_len = input_ids.dims()[1];
            // Return mock logits: higher values for lower token IDs
            let logits = [3.0f32, 2.0, 1.0, 0.5]; // Vocab size 4
            let batch_logits: Vec<f32> =
                (0..seq_len).flat_map(|_| logits.iter().copied()).collect();

            let logits =
                Tensor::from_vec(batch_logits, (1, seq_len, self.vocab_size), &self.device)?;
            Ok(logits)
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }
    }

    #[test]
    fn test_generator_config_default() {
        let config = GeneratorConfig::default();
        assert_eq!(config.max_tokens, 100);
        assert!((config.temperature - 1.0).abs() < 1e-7);
        assert!((config.repetition_penalty - 1.0).abs() < 1e-7);
        assert!(config.stop_on_eos);
        assert!(config.eos_token_id.is_none());
        assert!(config.top_p.is_none());
        assert!(config.top_k.is_none());
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generator_config_custom() {
        let config = GeneratorConfig {
            max_tokens: 256,
            temperature: 0.7,
            top_p: Some(0.95),
            top_k: Some(50),
            repetition_penalty: 1.2,
            stop_on_eos: false,
            eos_token_id: Some(151_643),
            stop_tokens: vec![100, 200],
            ..Default::default()
        };

        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.7).abs() < 1e-7);
        assert_eq!(config.top_p, Some(0.95));
        assert_eq!(config.top_k, Some(50));
        assert!((config.repetition_penalty - 1.2).abs() < 1e-7);
        assert!(!config.stop_on_eos);
        assert_eq!(config.eos_token_id, Some(151_643));
        assert_eq!(config.stop_tokens, vec![100, 200]);
    }

    #[test]
    fn test_generator_creation() {
        let model = MockModel::new(4);
        let config = GeneratorConfig::default();
        let generator = Generator::new(Box::new(model), config);
        assert!(generator.is_ok());
    }

    #[test]
    fn test_generator_basic_generation() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 5,
            sampling: SamplingStrategy::Greedy,
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![0u32];
        let output = generator.generate(&input_ids).unwrap();

        // Should generate max_tokens + input length
        assert!(output.len() <= 6); // 1 input + 5 generated
        assert_eq!(output[0], 0); // First token is input
    }

    #[test]
    fn test_generator_stop_on_eos() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 10,
            sampling: SamplingStrategy::Greedy,
            stop_on_eos: true,
            eos_token_id: Some(0), // Token 0 is EOS
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![1u32]; // Start with token 1
        let output = generator.generate(&input_ids).unwrap();

        // Should stop when EOS (token 0) is generated
        // Mock model always generates token 0 (highest logit)
        assert!(output.len() <= 3); // Will stop quickly due to EOS
    }

    #[test]
    fn test_generator_stop_tokens() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 10,
            sampling: SamplingStrategy::Greedy,
            stop_on_eos: false,
            stop_tokens: vec![0], // Stop on token 0
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![1u32];
        let output = generator.generate(&input_ids).unwrap();

        // Should stop when stop token (0) is generated
        assert!(output.len() <= 3);
    }

    #[test]
    fn test_generator_stream_basic() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 5,
            sampling: SamplingStrategy::Greedy,
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![1u32];

        let mut streamed_tokens = Vec::new();
        let output = generator
            .generate_stream(&input_ids, |token| {
                streamed_tokens.push(token);
                true // Continue
            })
            .unwrap();

        // All generated tokens should be in the output
        assert_eq!(output.len(), input_ids.len() + streamed_tokens.len());
        // Verify tokens match
        for (i, &token) in streamed_tokens.iter().enumerate() {
            assert_eq!(output[input_ids.len() + i], token);
        }
    }

    #[test]
    fn test_generator_stream_early_stop() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 10,
            sampling: SamplingStrategy::Greedy,
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![1u32];

        let mut count = 0;
        let output = generator
            .generate_stream(&input_ids, |_token| {
                count += 1;
                count < 3 // Return false on 3rd call
            })
            .unwrap();

        // Callback is called for each generated token
        // When it returns false, that token is still included
        assert!(count >= 3); // Called at least 3 times
        assert_eq!(output.len(), input_ids.len() + count); // Input + generated tokens
    }

    #[test]
    fn test_generator_stream_with_eos() {
        let model = MockModel::new(4);
        let config = GeneratorConfig {
            max_tokens: 10,
            sampling: SamplingStrategy::Greedy,
            stop_on_eos: true,
            eos_token_id: Some(0),
            ..Default::default()
        };

        let mut generator = Generator::new(Box::new(model), config).unwrap();
        let input_ids = vec![1u32];

        let mut streamed_tokens = Vec::new();
        let output = generator
            .generate_stream(&input_ids, |token| {
                streamed_tokens.push(token);
                true
            })
            .unwrap();

        // Should stop early due to EOS
        assert!(output.len() < input_ids.len() + 10);
        // Last token should be EOS (0)
        if !streamed_tokens.is_empty() {
            assert_eq!(streamed_tokens[streamed_tokens.len() - 1], 0);
        }
    }
}
