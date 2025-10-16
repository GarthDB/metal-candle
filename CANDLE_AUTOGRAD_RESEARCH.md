# Candle Autograd Research

**Goal**: Understand Candle's autograd API for implementing training loops

## What We Know

### VarBuilder (Found in our codebase)
- Used for initializing model weights
- From `candle_nn` package
- Usage: `VarBuilder::zeros(DType::F32, &device)`
- Creates weights via `.pp("layer_name")` calls

```rust
// Example from our code:
let vb = VarBuilder::zeros(DType::F16, &device);
let model = Qwen::new(&config, vb)?;
```

## ✅ FOUND: Candle Autograd API

### Core Types

1. **`Var`** - Trainable parameter wrapper
   - Located in `candle_core::Var`
   - Create with: `Var::from_tensor(&tensor)`
   - Access tensor: `var.as_tensor()`

2. **`GradStore`** - Gradient storage
   - Returned by `.backward()`
   - Access gradients: `grads.get(&var)`

### API Usage

```rust
use candle_core::{Tensor, Device, Var};

// 1. Create trainable parameter
let x_data = Tensor::new(&[2.0f32], &device)?;
let x = Var::from_tensor(&x_data)?;

// 2. Forward pass (use .as_tensor() to get the underlying tensor)
let y = x.as_tensor().sqr()?;

// 3. Backward pass (compute gradients)
let grads = y.backward()?;

// 4. Extract gradient for specific Var
if let Some(dx) = grads.get(&x) {
    // dx contains the gradient
}
```

### Verified Example

**Test**: `y = x²`, so `dy/dx = 2x`
- Input: `x = 2.0`
- Output: `y = 4.0`
- Gradient: `dy/dx = 4.0` ✅ (correct: 2 * 2 = 4)

## Key Insights

1. **Yes, Candle has automatic differentiation!**
2. **We use `Var` instead of `Tensor` for trainable parameters**
3. **Gradients are computed automatically with `.backward()`**
4. **Gradients are stored per-Var in `GradStore`**

## Application to LoRA

For our LoRA training:

```rust
// Current (non-trainable)
struct LoRALayer {
    lora_a: Tensor,  // ❌
    lora_b: Tensor,  // ❌
}

// Needed (trainable)
struct LoRALayer {
    lora_a: Var,  // ✅
    lora_b: Var,  // ✅
}

// Training step:
// 1. Forward: logits = model.forward(input)
// 2. Loss: loss = cross_entropy(logits, targets)
// 3. Backward: grads = loss.backward()
// 4. Update: optimizer.step(param, grad)
```

---

**Status**: ✅ Complete! Ready for implementation.

