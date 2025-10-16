//! Example: Loading a model from safetensors format
//!
//! This example demonstrates how to load a model using ModelLoader,
//! inspect its structure, and validate tensor shapes.
//!
//! Run with: cargo run --example load_model

use anyhow::Result;
use candle_core::DType;
use metal_candle::models::{ModelConfig, ModelLoader};
use metal_candle::Device;

fn main() -> Result<()> {
    println!("🚀 metal-candle Model Loading Example\n");

    // Create device with automatic fallback
    let device = Device::new_with_fallback(0);
    println!("📱 Device: {:?}\n", device.info());

    // Example 1: Loading a config file
    println!("📋 Example 1: Loading Model Configuration");
    println!("   (This would load from config.json in a real scenario)");

    let config_json = r#"{
        "architectures": ["qwen2"],
        "vocab_size": 32000,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "max_position_embeddings": 2048
    }"#;

    let config = ModelConfig::from_json(config_json)?;
    config.validate()?;

    println!("   ✓ Architecture: {:?}", config.architectures);
    println!("   ✓ Vocabulary size: {}", config.vocab_size);
    println!("   ✓ Hidden size: {}", config.hidden_size);
    println!("   ✓ Layers: {}", config.num_hidden_layers);
    println!("   ✓ Attention heads: {}", config.num_attention_heads);
    println!("   ✓ Head dimension: {}", config.head_dim());
    println!();

    // Example 2: Creating a model loader
    println!("🔧 Example 2: Creating Model Loader");

    let loader = ModelLoader::new(device.clone()).with_dtype(DType::F16);

    println!("   ✓ Device: {:?}", loader.device().info().device_type);
    println!("   ✓ Target dtype: {:?}", loader.dtype());
    println!();

    // Example 3: Inspecting a model (would need actual file)
    println!("🔍 Example 3: Inspecting Model Structure");
    println!("   (This would inspect an actual .safetensors file)");
    println!();

    // Create a demonstration of what loading would look like
    println!("   Example usage:");
    println!("   ```rust");
    println!("   let loader = ModelLoader::new(device)");
    println!("       .with_dtype(DType::F16);");
    println!();
    println!("   // Inspect without loading");
    println!("   let info = loader.inspect(\"model.safetensors\")?;");
    println!("   for (name, shape) in &info {{");
    println!("       println!(\"{{name}}: {{shape:?}}\");");
    println!("   }}");
    println!();
    println!("   // Load all tensors");
    println!("   let tensors = loader.load(\"model.safetensors\")?;");
    println!("   ```");
    println!();

    // Example 4: Validation
    println!("✅ Example 4: Model Validation");
    println!("   You can validate tensor shapes against expectations:");
    println!();
    println!("   ```rust");
    println!("   let mut expected = HashMap::new();");
    println!("   expected.insert(\"embed_tokens.weight\".to_string(),");
    println!("                  vec![config.vocab_size, config.hidden_size]);");
    println!();
    println!("   let tensors = loader.load_with_validation(");
    println!("       \"model.safetensors\",");
    println!("       &expected");
    println!("   )?;");
    println!("   ```");
    println!();

    println!("✨ Model loading infrastructure ready!");
    println!("   Next steps:");
    println!("   • Download a model in safetensors format");
    println!("   • Use ModelLoader to load it");
    println!("   • Build transformer architecture (Phase 2 continuation)");

    Ok(())
}
