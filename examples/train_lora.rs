//! Example of training a LoRA adapter.
//!
//! This example demonstrates how to:
//! 1. Set up a LoRA adapter for training
//! 2. Configure training hyperparameters
//! 3. Run the training loop
//! 4. Save checkpoints
//!
//! Run with: `cargo run --example train_lora`

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use metal_candle::training::{
    checkpoint::{save_checkpoint, CheckpointMetadata},
    AdamWConfig, LRScheduler, LoRAAdapter, LoRAAdapterConfig, TargetModule, Trainer,
    TrainingConfig,
};

fn main() -> Result<()> {
    println!("🚀 LoRA Training Example");
    println!("════════════════════════════════════════════════");
    println!();

    // 1. Setup device
    println!("📱 Setting up device...");
    let device = Device::Cpu; // Use CPU for this example
    println!("   Device: CPU");
    println!();

    // 2. Configure LoRA adapter
    println!("⚙️  Configuring LoRA adapter...");
    let lora_config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        dropout: 0.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
    };
    println!("   Rank: {}", lora_config.rank);
    println!("   Alpha: {}", lora_config.alpha);
    println!("   Target modules: Q-Proj, V-Proj");
    println!();

    // 3. Create mock model dimensions (Qwen2.5-Coder 0.5B)
    let hidden_size = 896;
    let intermediate_size = 4864;
    let num_layers = 24;

    println!("📊 Model configuration:");
    println!("   Hidden size: {hidden_size}");
    println!("   Intermediate size: {intermediate_size}");
    println!("   Number of layers: {num_layers}");
    println!();

    // 4. Create LoRA adapter
    println!("🔧 Creating LoRA adapter...");
    let lora_adapter = LoRAAdapter::new(
        hidden_size,
        intermediate_size,
        num_layers,
        &lora_config,
        &device,
    )?;

    let trainable_params = lora_adapter.num_trainable_parameters();
    // Estimate total model params for Qwen2.5-Coder 0.5B
    let total_params = 494_000_000; // Approximately 494M parameters
    let frozen_params = lora_adapter.num_frozen_parameters(total_params);
    println!("   Trainable parameters: {trainable_params:>10} (LoRA)");
    println!("   Frozen parameters:    {frozen_params:>10} (Base model)");
    println!(
        "   Trainable ratio:      {:>9.2}%",
        100.0 * trainable_params as f64 / (trainable_params + frozen_params) as f64
    );
    println!();

    // 5. Configure training
    println!("📚 Configuring training...");
    let training_config = TrainingConfig {
        num_epochs: 3,
        lr_scheduler: LRScheduler::WarmupCosine {
            warmup_steps: 100,
            max_lr: 1e-4,
            min_lr: 1e-6,
            total_steps: 1000,
        },
        optimizer_config: AdamWConfig {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        },
        max_grad_norm: Some(1.0),
    };
    println!("   Epochs: {}", training_config.num_epochs);
    println!(
        "   Max LR: {:.0e}",
        training_config.optimizer_config.learning_rate
    );
    println!("   Warmup steps: 100");
    println!("   Max grad norm: 1.0");
    println!();

    // 6. Create trainer
    println!("🎯 Creating trainer...");
    let mut trainer = Trainer::new(
        hidden_size,
        intermediate_size,
        num_layers,
        &lora_config,
        training_config,
        &device,
    )?;
    println!("   Trainer initialized");
    println!();

    // 7. Create dummy dataset (in real use, load from data)
    println!("📁 Preparing dataset...");
    let batch_size = 2;
    let seq_len = 128;
    let num_batches = 10;

    let mut dataset = Vec::new();
    for _ in 0..num_batches {
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let target_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        dataset.push((input_ids, target_ids));
    }
    println!("   Batches: {num_batches}");
    println!("   Batch size: {batch_size}");
    println!("   Sequence length: {seq_len}");
    println!();

    // 8. Define forward pass (mock for this example)
    println!("🔄 Defining forward pass...");
    let vocab_size = 32000;
    let forward_fn = |_input: &Tensor| -> metal_candle::Result<Tensor> {
        // In real training, this would be your model's forward pass
        // For this example, return random logits
        Ok(Tensor::randn(
            0f32,
            1f32,
            (batch_size, seq_len, vocab_size),
            &device,
        )?)
    };
    println!("   Vocab size: {vocab_size}");
    println!();

    // 9. Train!
    println!("🏋️  Training...");
    println!("════════════════════════════════════════════════");
    let metrics = trainer.train(&dataset, forward_fn)?;
    println!("════════════════════════════════════════════════");
    println!();

    // 10. Display results
    println!("📈 Training Results:");
    if let Some(first_metric) = metrics.first() {
        println!("   Initial loss: {:.4}", first_metric.loss);
    }
    if let Some(last_metric) = metrics.last() {
        println!("   Final loss:   {:.4}", last_metric.loss);
    }
    println!("   Total steps:  {}", metrics.len());
    println!();

    // 11. Save checkpoint
    println!("💾 Saving checkpoint...");
    let checkpoint_path = "lora_checkpoint.safetensors";
    let checkpoint_metadata = CheckpointMetadata {
        global_step: metrics.len(),
        loss: metrics.last().map(|m| m.loss).unwrap_or(0.0),
        learning_rate: metrics
            .last()
            .map(|m| m.learning_rate as f32)
            .unwrap_or(0.0),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    };

    save_checkpoint(
        trainer.lora_adapter(),
        checkpoint_path,
        Some(&checkpoint_metadata),
    )?;
    println!("   Saved to: {checkpoint_path}");
    println!();

    println!("✅ Training complete!");
    println!();
    println!("Next steps:");
    println!("  • Load the checkpoint for inference");
    println!("  • Merge LoRA weights into base model");
    println!("  • Evaluate on test set");

    Ok(())
}
