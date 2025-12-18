//! LoRA adapter hot-swapping demo.
//!
//! Demonstrates managing multiple LoRA adapters with the `AdapterRegistry`.
//!
//! This example shows:
//! - Loading multiple adapters
//! - Activating/deactivating adapters
//! - Switching between adapters
//! - Adapter memory management

use anyhow::Result;
use candle_core::Device;
use metal_candle::training::{AdapterRegistry, LoRAAdapter, LoRAAdapterConfig, TargetModule};

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

    // Demo 7: Real-world workflow
    println!("Demo 7: Real-World Workflow");
    println!("---------------------------");
    println!("In production, you would:");
    println!("1. Load base model once");
    println!("2. Load multiple task-specific adapters into registry");
    println!("3. Switch adapters based on user request/task");
    println!("4. No need to reload base model between switches");
    println!();

    println!("Example workflow:");
    println!("  User request: 'Write code' → Activate 'code-assistant'");
    println!("  User request: 'Chat' → Activate 'chat'");
    println!("  User request: 'Explain docs' → Activate 'docs'");
    println!();

    println!("=== Demo Complete ===");

    Ok(())
}
