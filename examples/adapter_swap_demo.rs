//! LoRA adapter hot-swapping demo.
//!
//! Demonstrates managing multiple LoRA adapters with the `AdapterRegistry`
//! and applying them to a model using the `ApplyAdapter` trait.
//!
//! This example shows:
//! - Loading multiple adapters
//! - Activating/deactivating adapters
//! - Switching between adapters
//! - Hot-swapping adapters on a live model
//! - Adapter memory management

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use metal_candle::models::{qwen::Qwen, ModelConfig};
use metal_candle::training::{
    AdapterRegistry, ApplyAdapter, LoRAAdapter, LoRAAdapterConfig, TargetModule,
};
use std::sync::Arc;

fn main() -> Result<()> {
    println!("=== Metal-Candle Adapter Hot-Swapping Demo ===\n");

    // Setup
    let device = Device::Cpu;
    let config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };

    // Create adapter registry
    let mut registry = AdapterRegistry::new();
    println!("Created adapter registry\n");

    // Demo 1: Loading multiple adapters
    println!("Demo 1: Loading Multiple Adapters");
    println!("----------------------------------");

    // Create adapters for different tasks
    let code_adapter = LoRAAdapter::new(768, 3072, 12, &config, &device)?;
    let chat_adapter = LoRAAdapter::new(768, 3072, 12, &config, &device)?;
    let docs_adapter = LoRAAdapter::new(768, 3072, 12, &config, &device)?;

    println!("Created 3 adapters:");
    println!(
        "  - code-assistant: {} params",
        code_adapter.num_trainable_parameters()
    );
    println!(
        "  - chat: {} params",
        chat_adapter.num_trainable_parameters()
    );
    println!(
        "  - docs: {} params",
        docs_adapter.num_trainable_parameters()
    );

    // Add to registry
    registry.add_adapter("code-assistant".to_string(), code_adapter)?;
    registry.add_adapter("chat".to_string(), chat_adapter)?;
    registry.add_adapter("docs".to_string(), docs_adapter)?;

    println!("\nAdded {} adapters to registry\n", registry.len());

    // Demo 2: Listing and inspecting adapters
    println!("Demo 2: Listing Adapters");
    println!("------------------------");
    let adapters = registry.list_adapters();
    println!("Available adapters:");
    for name in &adapters {
        println!("  - {}", name);
    }
    println!();

    // Demo 3: Activating adapters
    println!("Demo 3: Activating Adapters");
    println!("---------------------------");

    // Activate code assistant
    registry.activate("code-assistant")?;
    println!("Activated: {}", registry.active_adapter().unwrap());

    if let Some(adapter) = registry.get_active() {
        println!(
            "Active adapter has {} parameters",
            adapter.num_trainable_parameters()
        );
    }
    println!();

    // Demo 4: Switching adapters
    println!("Demo 4: Switching Adapters");
    println!("--------------------------");

    // Switch to chat adapter
    registry.activate("chat")?;
    println!("Switched to: {}", registry.active_adapter().unwrap());

    // Switch to docs adapter
    registry.activate("docs")?;
    println!("Switched to: {}", registry.active_adapter().unwrap());

    // Deactivate
    registry.deactivate();
    println!("Deactivated adapter");
    println!("Active adapter: {:?}\n", registry.active_adapter());

    // Demo 5: Unloading adapters
    println!("Demo 5: Unloading Adapters");
    println!("--------------------------");

    registry.unload_adapter("code-assistant")?;
    println!("Unloaded 'code-assistant'");
    println!("Remaining adapters: {}", registry.len());

    let remaining = registry.list_adapters();
    println!("Still available:");
    for name in &remaining {
        println!("  - {}", name);
    }
    println!();

    // Demo 6: Memory efficiency
    println!("Demo 6: Memory Efficiency");
    println!("-------------------------");
    println!("Note: The registry stores adapters without duplicating");
    println!("the base model weights. Each adapter only contains the");
    println!("low-rank matrices (A and B), making hot-swapping very");
    println!("memory efficient.");
    println!();

    // Calculate memory savings
    let adapter_params = if let Some(adapter) = registry.get_adapter("chat") {
        adapter.num_trainable_parameters()
    } else {
        0
    };

    // Typical base model has ~7B parameters
    // LoRA adapter has ~200k parameters (for rank=8)
    let base_model_params = 7_000_000_000u64;
    let memory_ratio = (adapter_params as f64) / (base_model_params as f64) * 100.0;

    println!("Adapter parameters: {}", adapter_params);
    println!("Base model parameters: ~{}", base_model_params);
    println!("Adapter is {:.3}% of base model size", memory_ratio);
    println!();

    // Demo 7: Hot-swapping with a model
    println!("Demo 7: Hot-Swapping with a Model");
    println!("----------------------------------");

    // Create a small test model
    let model_config = ModelConfig {
        architectures: vec!["qwen2".to_string()],
        vocab_size: 1000,
        hidden_size: 768,
        intermediate_size: 3072,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        num_key_value_heads: Some(6),
        max_position_embeddings: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        torch_dtype: Some("float32".to_string()),
    };

    let vb = VarBuilder::zeros(DType::F32, &device);
    let mut model = Qwen::new(&model_config, vb)?;
    println!("Created test Qwen model\n");

    // Create input
    let input_ids = Tensor::new(&[1u32, 2, 3, 4], &device)?.unsqueeze(0)?;

    // Run inference without adapter
    println!("Running inference without adapter...");
    let output_baseline = model.forward(&input_ids, None)?;
    println!("  Output shape: {:?}", output_baseline.dims());
    println!("  Has adapter: {}\n", model.has_adapter());

    // Apply chat adapter to model
    registry.activate("chat")?;
    if let Some(active) = registry.get_active() {
        println!("Applying 'chat' adapter to model...");
        model.apply_adapter(Arc::clone(active))?;
        println!("  Has adapter: {}", model.has_adapter());

        let output_chat = model.forward(&input_ids, None)?;
        println!("  Output shape: {:?}\n", output_chat.dims());
    }

    // Hot-swap to docs adapter
    registry.activate("docs")?;
    if let Some(active) = registry.get_active() {
        println!("Hot-swapping to 'docs' adapter...");
        model.apply_adapter(Arc::clone(active))?;
        println!("  Has adapter: {}", model.has_adapter());

        let output_docs = model.forward(&input_ids, None)?;
        println!("  Output shape: {:?}\n", output_docs.dims());
    }

    // Remove adapter
    println!("Removing adapter...");
    model.remove_adapter()?;
    println!("  Has adapter: {}", model.has_adapter());

    let output_no_adapter = model.forward(&input_ids, None)?;
    println!("  Output shape: {:?}\n", output_no_adapter.dims());

    // Demo 8: Real-world workflow
    println!("Demo 8: Real-World Workflow");
    println!("---------------------------");
    println!("In production, you would:");
    println!("1. Load base model once");
    println!("2. Load multiple task-specific adapters into registry");
    println!("3. Switch adapters based on user request/task using ApplyAdapter");
    println!("4. No need to reload base model between switches");
    println!("5. Adapter switching is instant (<2ms on Apple Silicon)");
    println!();

    println!("Example workflow:");
    println!("  User request: 'Write code'");
    println!("    → registry.activate('code-assistant')");
    println!("    → model.apply_adapter(registry.get_active()?)");
    println!("  User request: 'Chat'");
    println!("    → registry.activate('chat')");
    println!("    → model.apply_adapter(registry.get_active()?)");
    println!("  User request: 'Explain docs'");
    println!("    → registry.activate('docs')");
    println!("    → model.apply_adapter(registry.get_active()?)");
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
