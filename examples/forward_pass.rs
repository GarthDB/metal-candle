//! Example: Forward pass through Qwen2.5-Coder model
//!
//! This example demonstrates how to:
//! - Load a model configuration
//! - Initialize the Qwen model
//! - Perform a forward pass with input tokens
//!
//! Run with: `cargo run --example forward_pass`

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use metal_candle::models::{ModelConfig, Qwen};

fn main() -> Result<()> {
    println!("🚀 metal-candle Forward Pass Example\n");

    // 1. Set up device (using CPU for now - Metal has some compatibility issues to resolve)
    let device = Device::Cpu;
    println!("📱 Device: {device:?}\n");

    // 2. Create a sample configuration (in practice, load from config.json)
    let config = create_sample_config();
    println!("📋 Model Configuration:");
    println!("   ✓ Architecture: {:?}", config.architectures);
    println!("   ✓ Vocabulary size: {}", config.vocab_size);
    println!("   ✓ Hidden size: {}", config.hidden_size);
    println!("   ✓ Layers: {}", config.num_hidden_layers);
    println!("   ✓ Attention heads: {}", config.num_attention_heads);
    println!("   ✓ KV heads: {:?}", config.num_key_value_heads);
    println!(
        "   ✓ Max position embeddings: {}",
        config.max_position_embeddings
    );
    println!();

    // 3. Create model with zero-initialized weights (for demonstration)
    // In practice, you would load actual weights from a safetensors file
    println!("🔧 Initializing Model...");
    let vb = VarBuilder::zeros(DType::F16, &device);
    let model = Qwen::new(&config, vb)?;
    println!("   ✓ Model created with {} layers", model.num_layers());
    println!(
        "   ✓ Approximate parameters: ~{:.1}M",
        model.num_parameters() as f64 / 1_000_000.0
    );
    println!();

    // 4. Create sample input
    println!("📝 Creating Sample Input...");
    let batch_size = 2;
    let seq_len = 16;

    // In practice, these would come from tokenizing text
    // For demo, create random token IDs
    let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
    println!("   ✓ Input shape: [{batch_size}, {seq_len}]");
    println!("   ✓ (In practice, these would be tokenized text)");
    println!();

    // 5. Run forward pass
    println!("⚡ Running Forward Pass...");
    let logits = model.forward(&input_ids, None)?;
    let (b, s, v) = logits.dims3()?;
    println!("   ✓ Output logits shape: [{b}, {s}, {v}]");
    println!(
        "   ✓ Each position gets a probability distribution over {} tokens",
        v
    );
    println!();

    // 6. Show example of getting predictions
    println!("🎯 Example: Getting Next Token Predictions");
    println!("   To get the predicted next token:");
    println!("   1. Take logits for last position: logits[:, -1, :]");
    println!("   2. Apply softmax to get probabilities");
    println!("   3. Sample from distribution or take argmax");
    println!();

    println!("✨ Forward pass completed successfully!");
    println!();
    println!("💡 Next Steps:");
    println!("   • Load actual model weights from safetensors");
    println!("   • Use tokenizer to convert text to input_ids");
    println!("   • Implement generation loop with KV-caching");
    println!("   • Add temperature and top-p sampling");

    Ok(())
}

fn create_sample_config() -> ModelConfig {
    // This is a smaller version for demonstration
    // Real Qwen2.5-Coder models are much larger
    ModelConfig {
        architectures: vec!["Qwen2ForCausalLM".to_string()],
        vocab_size: 32_000,
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        num_key_value_heads: Some(4), // Grouped-query attention
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        torch_dtype: Some("float16".to_string()),
    }
}
