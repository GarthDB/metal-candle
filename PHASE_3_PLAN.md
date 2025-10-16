# Phase 3: LoRA Training Pipeline - Implementation Plan

**Status**: 🚧 In Progress  
**Branch**: `phase-3-lora-training`  
**Issue**: #3  
**Timeline**: Weeks 5-7 (Estimated 8-10 sessions)

## 📋 Overview

Phase 3 implements the complete training pipeline for LoRA (Low-Rank Adaptation) fine-tuning. This is the most complex phase, involving gradient computation, optimization, and checkpoint management.

## 🎯 Objectives

1. **LoRA Adapters** - Low-rank matrix decomposition for efficient fine-tuning
2. **Training Loop** - Forward/backward pass with loss computation
3. **AdamW Optimizer** - With learning rate scheduling
4. **Checkpoint Management** - Save/resume training state

## 📦 Deliverables Breakdown

### Part 1: LoRA Implementation (Sessions 1-3)

#### 1.1 LoRA Layer (`src/training/lora.rs`)
**Core component implementing low-rank adaptation**

```rust
pub struct LoRALayer {
    // Low-rank matrices: W + BA (where B·A is low-rank)
    lora_a: Tensor,  // (rank, in_features)
    lora_b: Tensor,  // (out_features, rank)
    
    // Hyperparameters
    rank: usize,
    alpha: f32,      // Scaling factor
    dropout: f32,
}
```

**Tasks**:
- [ ] Define LoRA layer struct
- [ ] Implement initialization (Gaussian for A, zeros for B)
- [ ] Add forward pass: `output = linear(x) + scale * (x @ A.T @ B.T)`
- [ ] Implement scaling: `scale = alpha / rank`
- [ ] Add dropout support (optional)
- [ ] Verify gradient flow through A and B matrices

**Tests**:
- [ ] Test initialization (A: Gaussian, B: zeros)
- [ ] Test forward pass output shape
- [ ] Test gradient computation (verify backprop works)
- [ ] Test scaling factor application

#### 1.2 LoRA Adapter (`src/training/lora.rs`)
**Apply LoRA to specific model layers**

```rust
pub struct LoRAAdapter {
    config: LoRAConfig,
    layers: HashMap<String, LoRALayer>,
    target_modules: Vec<String>,  // ["q_proj", "v_proj", etc.]
}
```

**Tasks**:
- [ ] Define adapter configuration
- [ ] Implement layer selection (which layers get LoRA)
- [ ] Add methods to inject LoRA into model
- [ ] Implement merge/unmerge (fold LoRA back into weights)
- [ ] Add parameter counting (trainable vs frozen)

**Tests**:
- [ ] Test adapter creation with different configs
- [ ] Test parameter filtering (only LoRA params trainable)
- [ ] Test merge/unmerge operations
- [ ] Test adapter application to model

---

### Part 2: Optimizer Implementation (Sessions 4-5)

#### 2.1 AdamW Optimizer (`src/training/optimizer.rs`)

```rust
pub struct AdamW {
    lr: f32,
    beta1: f32,      // Usually 0.9
    beta2: f32,      // Usually 0.999
    eps: f32,        // Usually 1e-8
    weight_decay: f32,
    
    // State for each parameter
    m: HashMap<String, Tensor>,  // First moment
    v: HashMap<String, Tensor>,  // Second moment
    step: usize,
}
```

**Tasks**:
- [ ] Implement AdamW algorithm
- [ ] Add bias correction for moments
- [ ] Implement weight decay (decoupled from gradient)
- [ ] Add gradient clipping option
- [ ] Zero gradient functionality
- [ ] State management (m, v tensors)

**Tests**:
- [ ] Test basic parameter update
- [ ] Test bias correction
- [ ] Test weight decay application
- [ ] Test gradient clipping
- [ ] Test state persistence

#### 2.2 Learning Rate Scheduling (`src/training/scheduler.rs`)

```rust
pub enum LRScheduler {
    Constant(f32),
    Linear { start: f32, end: f32, steps: usize },
    Cosine { max_lr: f32, min_lr: f32, total_steps: usize },
    WarmupCosine { warmup_steps: usize, total_steps: usize, max_lr: f32 },
}
```

**Tasks**:
- [ ] Implement constant LR
- [ ] Implement linear warmup
- [ ] Implement cosine annealing
- [ ] Implement warmup + cosine (most common)
- [ ] Add `get_lr(step)` method

**Tests**:
- [ ] Test each schedule type
- [ ] Test warmup phase
- [ ] Test cosine decay
- [ ] Test boundary conditions

---

### Part 3: Training Loop (Sessions 6-8)

#### 3.1 Trainer (`src/training/trainer.rs`)

```rust
pub struct Trainer {
    model: Qwen,
    lora_adapter: LoRAAdapter,
    optimizer: AdamW,
    scheduler: LRScheduler,
    config: TrainingConfig,
}

pub struct TrainingConfig {
    learning_rate: f32,
    batch_size: usize,
    num_epochs: usize,
    gradient_accumulation_steps: usize,
    max_grad_norm: f32,
    warmup_steps: usize,
    eval_steps: usize,
}
```

**Tasks**:
- [ ] Implement training loop structure
- [ ] Add forward pass with loss computation
- [ ] Implement backward pass (gradient computation)
- [ ] Add gradient accumulation
- [ ] Implement evaluation loop
- [ ] Add progress tracking and logging
- [ ] Memory efficient batch processing
- [ ] Early stopping support

**Tests**:
- [ ] Test single training step
- [ ] Test gradient accumulation
- [ ] Test loss computation
- [ ] Test evaluation mode
- [ ] Test early stopping

#### 3.2 Loss Functions (`src/training/loss.rs`)

```rust
pub fn cross_entropy_loss(
    logits: &Tensor,        // (batch, seq_len, vocab_size)
    targets: &Tensor,       // (batch, seq_len)
    ignore_index: Option<u32>,
) -> Result<Tensor>
```

**Tasks**:
- [ ] Implement cross-entropy loss
- [ ] Add label smoothing option
- [ ] Support ignore_index for padding
- [ ] Add per-token vs per-sequence loss
- [ ] Numerically stable implementation

**Tests**:
- [ ] Test basic cross-entropy
- [ ] Test with ignore_index
- [ ] Test label smoothing
- [ ] Test numerical stability

---

### Part 4: Checkpoint Management (Sessions 9-10)

#### 4.1 Checkpoint Manager (`src/checkpoint/manager.rs`)

```rust
pub struct CheckpointManager {
    save_dir: PathBuf,
    keep_last: usize,    // Number of checkpoints to keep
}

pub struct Checkpoint {
    lora_weights: HashMap<String, Tensor>,
    optimizer_state: OptimizerState,
    metadata: CheckpointMetadata,
}

pub struct CheckpointMetadata {
    epoch: usize,
    step: usize,
    loss: f32,
    learning_rate: f32,
    config: LoRAConfig,
    timestamp: String,
}
```

**Tasks**:
- [ ] Implement save checkpoint
- [ ] Implement load checkpoint
- [ ] Add metadata serialization
- [ ] Implement checkpoint rotation (keep last N)
- [ ] Add resume training support
- [ ] Safetensors format for weights

**Tests**:
- [ ] Test save/load roundtrip
- [ ] Test metadata persistence
- [ ] Test checkpoint rotation
- [ ] Test resume from checkpoint
- [ ] Test format compatibility

---

### Part 5: Integration & Examples (Session 11-12)

#### 5.1 Training Example (`examples/train_lora.rs`)

```rust
// Complete end-to-end training example
fn main() -> Result<()> {
    // 1. Load base model
    let model = ModelLoader::new(device).load("model.safetensors")?;
    
    // 2. Create LoRA adapter
    let lora_config = LoRAConfig {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".into(), "v_proj".into()],
        ..Default::default()
    };
    let adapter = LoRAAdapter::new(model, lora_config)?;
    
    // 3. Setup training
    let config = TrainingConfig::default();
    let trainer = Trainer::new(adapter, config)?;
    
    // 4. Train
    let dataset = load_dataset("data.json")?;
    trainer.train(dataset)?;
    
    // 5. Save checkpoint
    trainer.save_checkpoint("output/lora_checkpoint")?;
}
```

**Tasks**:
- [ ] Create working training example
- [ ] Add dataset loading utilities
- [ ] Show checkpoint save/load
- [ ] Add progress reporting
- [ ] Document all parameters

#### 5.2 Integration Tests (`tests/training/`)

**Tests**:
- [ ] End-to-end training test (small model)
- [ ] Checkpoint save/load/resume test
- [ ] Gradient flow verification
- [ ] Loss convergence test (overfitting small dataset)
- [ ] Memory usage test

---

## 🧪 Testing Strategy

### Unit Tests (Target: 80%+ coverage)
- [ ] LoRA layer operations
- [ ] Optimizer updates
- [ ] Loss computation
- [ ] Checkpoint I/O
- [ ] Scheduler behavior

### Integration Tests
- [ ] Full training loop
- [ ] Checkpoint resume
- [ ] Gradient flow
- [ ] Memory efficiency

### Property Tests
- [ ] Gradient computation correctness
- [ ] Optimizer convergence
- [ ] Numerical stability

## 📚 Documentation Requirements

- [ ] API docs for all public types
- [ ] Training guide in docs/
- [ ] Hyperparameter tuning guide
- [ ] Checkpoint format specification
- [ ] Example walkthroughs

## 🎯 Success Criteria

### Functional
- [ ] LoRA adapters apply to model successfully
- [ ] Training loop completes without errors
- [ ] Loss decreases over training
- [ ] Checkpoints save and load correctly
- [ ] Can resume training from checkpoint

### Quality
- [ ] ≥80% code coverage
- [ ] Zero clippy warnings
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Example working end-to-end

### Performance
- [ ] Training throughput reasonable (to be benchmarked in Phase 5)
- [ ] Memory usage ≤ MLX baseline
- [ ] Gradient computation correct (verified with finite differences)

## 📊 Progress Tracking

**Sessions Estimated**: 10-12 focused sessions
- LoRA Implementation: 3 sessions
- Optimizer: 2 sessions
- Training Loop: 3 sessions
- Checkpoints: 2 sessions
- Integration/Testing: 2 sessions

**Current Status**: Ready to begin ✅

## 🔗 Dependencies

**Requires (from Phase 2)**:
- ✅ Model loading infrastructure
- ✅ Qwen architecture
- ✅ Transformer components

**Blocks**:
- Phase 4: Inference & Generation
- Phase 5: Quality & Benchmarking

## 📝 Notes

### LoRA Paper Reference
- Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- Key insight: Most gradient updates during fine-tuning are low-rank
- Typical rank: 4-16 (much smaller than hidden dimension)
- Alpha: Usually 2×rank for stable training

### Candle Specifics
- Need to work with Candle's gradient tape/backward API
- May need to implement custom backward passes
- Metal backend should work automatically via Candle

### Potential Challenges
1. **Gradient Computation**: Candle's autograd API
2. **Memory Management**: Large model + gradients + optimizer state
3. **Numerical Stability**: Loss computation, gradient clipping
4. **State Management**: Tracking training progress across steps

---

**Ready to begin Phase 3!** Let's start with Part 1: LoRA Implementation. 🚀

