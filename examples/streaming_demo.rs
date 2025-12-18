//! Streaming inference demo.
//!
//! Demonstrates real-time token generation with the streaming API.
//!
//! This example shows:
//! - Sync streaming with callbacks
//! - Async streaming with futures
//! - Rich token metadata (probability, logit, etc.)
//! - Early stopping and cancellation

use anyhow::Result;
use candle_core::{Device, Tensor};
use metal_candle::error::Result as MetalResult;
use metal_candle::inference::{Generator, GeneratorConfig, SamplingStrategy, StreamToken};
use metal_candle::models::LanguageModel;

/// Mock model for demonstration purposes
struct DemoModel {
    device: Device,
    vocab_size: usize,
}

impl DemoModel {
    fn new() -> Self {
        Self {
            device: Device::Cpu,
            vocab_size: 100,
        }
    }
}

impl LanguageModel for DemoModel {
    fn forward(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> MetalResult<Tensor> {
        let seq_len = input_ids.dims()[1];
        // Generate mock logits that favor lower token IDs
        let logits: Vec<f32> = (0..seq_len)
            .flat_map(|_| {
                (0..self.vocab_size)
                    .map(|i| 5.0 - (i as f32 * 0.05))
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

fn main() -> Result<()> {
    println!("=== Metal-Candle Streaming Inference Demo ===\n");

    // Create model and generator
    let model = DemoModel::new();
    let config = GeneratorConfig {
        max_tokens: 10,
        sampling: SamplingStrategy::Greedy,
        temperature: 1.0,
        repetition_penalty: 1.0,
        stop_on_eos: false,
        eos_token_id: None,
        ..Default::default()
    };

    let mut generator = Generator::new(Box::new(model), config)?;

    // Demo 1: Sync streaming with callback
    println!("Demo 1: Sync Streaming with Callback");
    println!("--------------------------------------");
    let input_ids = vec![1u32, 2, 3];
    let mut token_count = 0;

    generator.generate_stream(&input_ids, |token: StreamToken| {
        token_count += 1;
        println!(
            "Token {}: id={}, prob={:.2}%, logit={:.2}",
            token_count,
            token.token_id,
            token.probability * 100.0,
            token.logit
        );
        true // Continue generation
    })?;

    println!("\nGenerated {} tokens\n", token_count);

    // Demo 2: Early stopping
    println!("Demo 2: Early Stopping");
    println!("----------------------");
    let mut stop_count = 0;
    generator.generate_stream(&input_ids, |token: StreamToken| {
        stop_count += 1;
        println!("Token {}: id={}", stop_count, token.token_id);
        stop_count < 5 // Stop after 5 tokens
    })?;

    println!("\nStopped early after {} tokens\n", stop_count);

    // Demo 3: Async streaming (requires 'streaming' feature)
    #[cfg(feature = "streaming")]
    {
        println!("Demo 3: Async Streaming");
        println!("-----------------------");
        tokio::runtime::Runtime::new()?.block_on(async {
            use futures::pin_mut;
            use futures::stream::StreamExt;

            let stream = generator.generate_stream_async(&input_ids);
            pin_mut!(stream);

            let mut async_count = 0;
            while let Some(result) = stream.next().await {
                match result {
                    Ok(token) => {
                        async_count += 1;
                        println!(
                            "Async token {}: id={}, prob={:.2}%",
                            async_count,
                            token.token_id,
                            token.probability * 100.0
                        );
                    }
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        break;
                    }
                }
            }

            println!("\nAsync generated {} tokens\n", async_count);
        });
    }

    #[cfg(not(feature = "streaming"))]
    {
        println!("Demo 3: Async Streaming");
        println!("-----------------------");
        println!("(Async streaming requires 'streaming' feature)");
        println!("Run with: cargo run --example streaming_demo --features streaming\n");
    }

    println!("=== Demo Complete ===");

    Ok(())
}
