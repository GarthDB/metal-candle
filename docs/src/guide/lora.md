# LoRA Training

Low-Rank Adaptation (LoRA) for efficient fine-tuning of transformer models.

## What is LoRA?

LoRA is a parameter-efficient fine-tuning technique that:
- **Freezes** pre-trained model weights
- **Trains** small adapter matrices (rank << hidden_size)
- **Achieves** comparable results to full fine-tuning
- **Uses** ~0.1% of trainable parameters

## Quick Start

```rust
use metal_candle::training::{
    LoRAAdapter, LoRAAdapterConfig, TargetModule,
    Trainer, TrainingConfig, LRScheduler, AdamWConfig
};

// 1. Create LoRA adapter
let lora_config = LoRAAdapterConfig {
    rank: 8,
    alpha: 16.0,
    dropout: 0.0,
    target_modules: vec![
        TargetModule::QProj,
        TargetModule::VProj,
    ],
};

let adapter = LoRAAdapter::new(&model, lora_config, &device)?;

// 2. Configure training
let training_config = TrainingConfig {
    num_epochs: 3,
    lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
    optimizer_config: AdamWConfig::default(),
    max_grad_norm: Some(1.0),
};

// 3. Train
let trainer = Trainer::new(adapter, training_config)?;
let metrics = trainer.train(&dataset)?;

// 4. Save checkpoint
save_checkpoint(&trainer.lora_adapter(), "checkpoint.safetensors", None)?;
```

## LoRA Configuration

### Rank (r)

The rank determines adapter size and capacity:

```rust
LoRAAdapterConfig {
    rank: 8,  // Lower = fewer parameters, faster training
    // rank: 16,  // Higher = more capacity, better results
    // rank: 32,  // Very high = approaching full fine-tuning
    ..Default::default()
}
```

**Guidelines:**
- **r=4-8**: Simple tasks, small datasets
- **r=16**: General purpose, good balance
- **r=32-64**: Complex tasks, large datasets

### Alpha (α)

Scaling factor for LoRA weights:

```rust
LoRAAdapterConfig {
    rank: 8,
    alpha: 16.0,  // Typically 2x rank
    ..Default::default()
}
```

**Common settings:**
- alpha = rank (conservative)
- alpha = 2 * rank (recommended)
- alpha = 4 * rank (aggressive)

### Target Modules

Which layers to adapt:

```rust
use metal_candle::training::TargetModule;

LoRAAdapterConfig {
    target_modules: vec![
        TargetModule::QProj,   // Query projection
        TargetModule::VProj,   // Value projection
        // TargetModule::KProj,   // Key projection (optional)
        // TargetModule::OProj,   // Output projection (optional)
    ],
    ..Default::default()
}
```

**Recommendations:**
- **Q+V**: Most common, good results (default)
- **Q+K+V+O**: Maximum capacity
- **Q only**: Minimal, fastest

### Dropout

Regularization (usually 0.0 for LoRA):

```rust
LoRAAdapterConfig {
    dropout: 0.0,  // No dropout (recommended for LoRA)
    // dropout: 0.1,  // Light regularization
    ..Default::default()
}
```

## Training Configuration

### Learning Rate Schedule

```rust
use metal_candle::training::LRScheduler;

// Warmup + Cosine decay (recommended)
let scheduler = LRScheduler::warmup_cosine(
    100,    // warmup_steps
    1000,   // total_steps
    1e-4,   // max_lr
    1e-6,   // min_lr
);

// Constant learning rate
let scheduler = LRScheduler::constant(1e-4);

// Linear decay
let scheduler = LRScheduler::linear(1000, 1e-4, 1e-6);
```

### Optimizer

```rust
use metal_candle::training::AdamWConfig;

let optimizer_config = AdamWConfig {
    learning_rate: 1e-4,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weight_decay: 0.01,
};
```

### Gradient Clipping

Prevent exploding gradients:

```rust
TrainingConfig {
    max_grad_norm: Some(1.0),  // Clip to max norm of 1.0
    ..Default::default()
}
```

## Training Loop

```rust
// Create trainer
let trainer = Trainer::new(adapter, config)?;

// Train (automatic loop)
let metrics = trainer.train(&dataset)?;

// Metrics
println!("Final loss: {:.4}", metrics.final_loss);
println!("Epochs: {}", metrics.epochs_completed);
```

## Checkpointing

### Save Checkpoint

```rust
use metal_candle::training::save_checkpoint;

// Save with metadata
let metadata = Some(hashmap! {
    "task" => "code-generation",
    "dataset" => "my-dataset-v1",
});

save_checkpoint(
    &trainer.lora_adapter(),
    "checkpoint.safetensors",
    metadata,
)?;
```

### Load Checkpoint

```rust
use metal_candle::training::load_checkpoint;

// Load LoRA weights
let (lora_weights, metadata) = load_checkpoint(
    "checkpoint.safetensors",
    &device,
)?;

// Apply to new adapter
let adapter = LoRAAdapter::new(&model, lora_config, &device)?;
adapter.load_weights(lora_weights)?;
```

## Performance

### metal-candle Characteristics

LoRA operations leverage Metal GPU acceleration:

| Operation | Metal GPU | CPU | GPU Speedup |
|-----------|-------------|-----|---------|
| Small (512×512, r=8) | 37.0 µs | 65.0 µs | **1.76x** |
| Medium (1024×1024, r=8) | 54.8 µs | 125.6 µs | **2.29x** |
| Large (2048×2048, r=8) | 98.4 µs | 262.3 µs | **2.67x** |

**Value Proposition**: While raw throughput is optimized for ergonomics, metal-candle excels in type safety, single-binary deployment, and production quality.

See [Benchmarks](../testing/benchmarks.md) for complete metrics.

### Tips for Efficient Training

1. **Use Metal GPU for acceleration:**
```rust
let device = Device::new_metal(0)?;
```

2. **Use F16 precision to reduce memory:**
```rust
let device = Device::new_metal(0)?;
let loader = ModelLoader::new(device)
    .with_dtype(DType::F16);
```

3. **Start with lower rank for iteration:**
```rust
// Start with r=8, increase if needed
LoRAAdapterConfig { rank: 8, ..Default::default() }
```

4. **Leverage type safety:**
```rust
// Rust's compiler catches errors at compile time
// No runtime surprises from shape mismatches
```

## Example: Complete Training Script

```rust
use metal_candle::*;
use metal_candle::models::{Qwen, ModelConfig, ModelLoader};
use candle_nn::VarBuilder;
use anyhow::Result;

fn main() -> Result<()> {
    // Setup device
    let device = Device::new_metal(0)?;
    
    // Load model
    let config = ModelConfig::from_file("config.json")?;
    let loader = ModelLoader::new(device.clone());
    let tensors = loader.load("model.safetensors")?;
    let vb = VarBuilder::from_tensors(tensors, candle_core::DType::F16, &device);
    let model = Qwen::new(&config, vb)?;
    
    // Create LoRA adapter (r=8, alpha=16)
    let lora_config = LoRAAdapterConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec![TargetModule::QProj, TargetModule::VProj],
        ..Default::default()
    };
    let adapter = LoRAAdapter::new(&model, lora_config, &device)?;
    
    // Training configuration
    let config = TrainingConfig {
        num_epochs: 3,
        lr_scheduler: LRScheduler::warmup_cosine(100, 1000, 1e-4, 1e-6),
        optimizer_config: AdamWConfig::default(),
        max_grad_norm: Some(1.0),
    };
    
    // Train
    let trainer = Trainer::new(adapter, config)?;
    let metrics = trainer.train(&dataset)?;
    
    // Save
    save_checkpoint(&trainer.lora_adapter(), "lora_weights.safetensors", None)?;
    
    println!("✅ Training complete!");
    println!("Final loss: {:.4}", metrics.final_loss);
    
    Ok(())
}
```

## See Also

- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Original research
- [Model Loading](./models.md) - Loading base models
- [Benchmarks](../testing/benchmarks.md) - Performance data
