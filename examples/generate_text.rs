//! Text generation example using the Generator API.
//!
//! This example demonstrates:
//! 1. Setting up a language model for generation
//! 2. Configuring different sampling strategies
//! 3. Basic token generation
//! 4. Streaming generation with callbacks
//! 5. Using repetition penalty
//! 6. Stop conditions (EOS tokens, custom stop tokens)
//!
//! Run with: `cargo run --example generate_text --features custom-metal`

use anyhow::Result;
use candle_core::{Device, Tensor};
use metal_candle::inference::{Generator, GeneratorConfig, SamplingStrategy};
use metal_candle::models::LanguageModel;

/// Simple mock model for demonstration purposes.
///
/// In a real application, you would use a proper model like Qwen loaded from safetensors:
/// ```no_run
/// use metal_candle::models::{ModelConfig, Qwen};
/// use candle_nn::VarBuilder;
///
/// let config = ModelConfig::from_file("config.json")?;
/// let vb = VarBuilder::from_safetensors(...);
/// let model = Qwen::new(&config, vb)?;
/// ```
struct DemoModel {
    device: Device,
    vocab_size: usize,
}

impl DemoModel {
    fn new(vocab_size: usize, device: Device) -> Self {
        Self { device, vocab_size }
    }
}

impl LanguageModel for DemoModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>,
    ) -> metal_candle::error::Result<Tensor> {
        let seq_len = input_ids.dims()[1];

        // Simple mock: return logits that prefer lower token IDs
        // In a real model, this would be actual transformer forward pass
        let mut logits_vec = Vec::new();
        for _ in 0..seq_len {
            for i in 0..self.vocab_size {
                // Decreasing logits: token 0 has highest, etc.
                let logit = (self.vocab_size - i) as f32;
                logits_vec.push(logit);
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

fn main() -> Result<()> {
    println!("🎯 Metal-Candle Text Generation Demo");
    println!("════════════════════════════════════════════════");
    println!();

    // Setup device
    let device = Device::Cpu;
    println!("Device: CPU");
    println!();

    // Create model
    let vocab_size = 100;
    let model = Box::new(DemoModel::new(vocab_size, device));

    // Demo 1: Basic Greedy Generation
    demo_greedy_generation(model.as_ref())?;

    // Demo 2: Different Sampling Strategies
    demo_sampling_strategies()?;

    // Demo 3: Streaming Generation
    demo_streaming_generation()?;

    // Demo 4: Repetition Penalty
    demo_repetition_penalty()?;

    // Demo 5: Stop Conditions
    demo_stop_conditions()?;

    println!("✅ All demos completed successfully!");
    println!();
    println!("Next steps:");
    println!("  1. Load a real model (Qwen, LLaMA, etc.) from safetensors");
    println!("  2. Use a proper tokenizer for input/output");
    println!("  3. Experiment with different sampling parameters");
    println!("  4. Try streaming for real-time generation");

    Ok(())
}

fn demo_greedy_generation(_model: &dyn LanguageModel) -> Result<()> {
    println!("📝 Demo 1: Basic Greedy Generation");
    println!("────────────────────────────────────────────────");

    let config = GeneratorConfig {
        max_tokens: 10,
        sampling: SamplingStrategy::Greedy,
        repetition_penalty: 1.0,
        stop_on_eos: false,
        ..Default::default()
    };

    let device = Device::Cpu;
    let model_box = Box::new(DemoModel::new(100, device));
    let mut generator = Generator::new(model_box, config)?;

    let input_ids = vec![5u32, 10, 15];
    println!("Input tokens: {:?}", input_ids);

    let output = generator.generate(&input_ids)?;
    println!("Generated tokens: {:?}", output);
    println!("Total length: {} tokens", output.len());
    println!();

    Ok(())
}

fn demo_sampling_strategies() -> Result<()> {
    println!("🎲 Demo 2: Different Sampling Strategies");
    println!("────────────────────────────────────────────────");

    let device = Device::Cpu;
    let strategies = vec![
        ("Greedy (deterministic)", SamplingStrategy::Greedy),
        ("Top-k (k=10)", SamplingStrategy::TopK { k: 10 }),
        (
            "Top-p (nucleus, p=0.95)",
            SamplingStrategy::TopP { p: 0.95 },
        ),
        (
            "Temperature (T=0.7)",
            SamplingStrategy::Temperature { temperature: 0.7 },
        ),
    ];

    for (name, strategy) in strategies {
        println!("{name}:");

        let config = GeneratorConfig {
            max_tokens: 5,
            sampling: strategy,
            repetition_penalty: 1.0,
            ..Default::default()
        };

        let model = Box::new(DemoModel::new(100, device.clone()));
        let mut generator = Generator::new(model, config)?;

        let input_ids = vec![1u32];
        let output = generator.generate(&input_ids)?;
        println!("  Generated: {:?}", output);
    }

    println!();
    Ok(())
}

fn demo_streaming_generation() -> Result<()> {
    println!("🌊 Demo 3: Streaming Generation");
    println!("────────────────────────────────────────────────");
    println!("Simulating real-time token-by-token generation...");
    println!();

    let device = Device::Cpu;
    let config = GeneratorConfig {
        max_tokens: 10,
        sampling: SamplingStrategy::Greedy,
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device));
    let mut generator = Generator::new(model, config)?;

    let input_ids = vec![1u32, 2];
    print!("Tokens: ");

    let mut token_count = 0;
    let output = generator.generate_stream(&input_ids, |token| {
        print!("{} ", token.token_id);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        token_count += 1;

        // Could stop early based on condition
        true // Continue generation
    })?;

    println!();
    println!();
    println!("Total tokens generated: {token_count}");
    println!("Full sequence: {:?}", output);
    println!();

    Ok(())
}

fn demo_repetition_penalty() -> Result<()> {
    println!("🔁 Demo 4: Repetition Penalty");
    println!("────────────────────────────────────────────────");

    let device = Device::Cpu;

    // Without penalty
    println!("Without repetition penalty (1.0):");
    let config = GeneratorConfig {
        max_tokens: 8,
        sampling: SamplingStrategy::Greedy,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device.clone()));
    let mut generator = Generator::new(model, config)?;
    let output = generator.generate(&[1u32])?;
    println!("  {:?}", output);

    // With penalty
    println!("With repetition penalty (1.5):");
    let config = GeneratorConfig {
        max_tokens: 8,
        sampling: SamplingStrategy::Greedy,
        repetition_penalty: 1.5,
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device));
    let mut generator = Generator::new(model, config)?;
    let output = generator.generate(&[1u32])?;
    println!("  {:?}", output);
    println!();

    println!("Note: Higher penalty (>1.0) reduces repetition");
    println!();

    Ok(())
}

fn demo_stop_conditions() -> Result<()> {
    println!("🛑 Demo 5: Stop Conditions");
    println!("────────────────────────────────────────────────");

    let device = Device::Cpu;

    // EOS token
    println!("With EOS token (stop_on_eos=true, eos_token_id=0):");
    let config = GeneratorConfig {
        max_tokens: 20,
        sampling: SamplingStrategy::Greedy,
        stop_on_eos: true,
        eos_token_id: Some(0),
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device.clone()));
    let mut generator = Generator::new(model, config)?;
    let output = generator.generate(&[5u32])?;
    println!("  Generated {} tokens (stopped at EOS)", output.len());
    println!("  {:?}", output);
    println!();

    // Custom stop tokens
    println!("With custom stop tokens [0, 1, 2]:");
    let config = GeneratorConfig {
        max_tokens: 20,
        sampling: SamplingStrategy::Greedy,
        stop_on_eos: false,
        stop_tokens: vec![0, 1, 2],
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device.clone()));
    let mut generator = Generator::new(model, config)?;
    let output = generator.generate(&[10u32])?;
    println!(
        "  Generated {} tokens (stopped at stop token)",
        output.len()
    );
    println!("  {:?}", output);
    println!();

    // Streaming with early stop
    println!("Streaming with callback-based early stop:");
    let config = GeneratorConfig {
        max_tokens: 20,
        sampling: SamplingStrategy::Greedy,
        ..Default::default()
    };

    let model = Box::new(DemoModel::new(100, device));
    let mut generator = Generator::new(model, config)?;

    let mut count = 0;
    let output = generator.generate_stream(&[10u32], |_token| {
        count += 1;
        count < 5 // Stop after 5 tokens
    })?;

    println!(
        "  Generated {} tokens (stopped by callback)",
        output.len() - 1
    );
    println!("  {:?}", output);
    println!();

    Ok(())
}
