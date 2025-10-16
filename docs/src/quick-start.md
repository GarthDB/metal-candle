# Quick Start

Get started with metal-candle in minutes!

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust 1.70+ (`rustup update`)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
metal-candle = "0.1"
candle-core = "0.9"
```

## Your First Example

```rust
use metal_candle::{Device, TensorExt};
use candle_core::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create device with Metal acceleration
    let device = Device::new_with_fallback(0);
    
    println!("Using device: {:?}", device.info());
    
    // Create a tensor
    let tensor = Tensor::randn(0f32, 1f32, (2, 512), device.as_ref())?;
    
    // Apply numerically stable softmax
    let probs = tensor.softmax_stable()?;
    
    println!("Probabilities shape: {:?}", probs.shape());
    
    Ok(())
}
```

## Run It

```bash
cargo run
```

## Next Steps

- [Device Management](./guide/devices.md) - Learn about Metal device handling
- [Tensor Operations](./guide/tensors.md) - Explore available operations
- [Examples](https://github.com/GarthDB/metal-candle/tree/main/examples) - See more examples

## Getting Help

- [Troubleshooting](./reference/troubleshooting.md)
- [FAQ](./reference/faq.md)
- [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
