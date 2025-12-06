# Installation

Getting metal-candle set up on your Apple Silicon Mac.

## Prerequisites

### Required

- **Apple Silicon Mac**: M1, M2, M3, or M4 chip
- **macOS**: 12.0 (Monterey) or later
- **Rust**: 1.75 or later

### Check Your System

```bash
# Check macOS version
sw_vers

# Check if you have Apple Silicon
sysctl -n macintosh.cpu.brand_string

# Check Rust version
rustc --version
```

## Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
```

Verify installation:
```bash
rustc --version  # Should be 1.75 or higher
cargo --version
```

## Add metal-candle to Your Project

### From crates.io (Recommended)

```toml
[dependencies]
metal-candle = "1.0"
```

### From Git (Latest)

```toml
[dependencies]
metal-candle = { git = "https://github.com/GarthDB/metal-candle", tag = "v1.0.0" }
```

### With Features

```toml
[dependencies]
metal-candle = { version = "1.0", features = ["embeddings"] }
```

**Available features:**
- `embeddings`: Sentence-transformers support (E5, MiniLM, MPNet)
- `metal`: Metal GPU support (enabled by default)

## Verify Installation

Create a test project:

```bash
cargo new test-metal-candle
cd test-metal-candle
```

Add dependency in `Cargo.toml`:
```toml
[dependencies]
metal-candle = "1.0"
anyhow = "1.0"
```

Test with a simple program (`src/main.rs`):
```rust
use metal_candle::backend::Device;
use anyhow::Result;

fn main() -> Result<()> {
    let device = Device::new_with_fallback(0);
    let info = device.info();
    
    println!("✅ metal-candle installed successfully!");
    println!("Device: {:?}", info.device_type);
    println!("Metal available: {}", info.metal_available);
    
    Ok(())
}
```

Run it:
```bash
cargo run
```

Expected output:
```
✅ metal-candle installed successfully!
Device: Metal
Metal available: true
```

## Troubleshooting

### "failed to resolve: use of undeclared crate or module"

- Check `Cargo.toml` has `metal-candle` dependency
- Run `cargo build` to fetch dependencies
- Try `cargo clean && cargo build`

### "Metal device not available"

If you see `Device: Cpu` instead of `Device: Metal`:
- Verify you're on Apple Silicon (not Intel Mac)
- Check macOS version is 12.0+
- Restart your Mac

This is OK for testing, but Metal GPU won't be used.

### Build Errors

**Error**: "requires rustc 1.75 or higher"
```bash
rustup update stable
rustc --version  # Verify 1.75+
```

**Error**: Candle compilation failures
- Ensure Xcode Command Line Tools: `xcode-select --install`
- Try: `cargo clean && cargo build`

### Slow First Build

The first build downloads and compiles dependencies (especially Candle). This is normal and can take 5-10 minutes. Subsequent builds will be much faster.

## Development Setup

For contributing to metal-candle:

```bash
# Clone repository
git clone https://github.com/GarthDB/metal-candle
cd metal-candle

# Install additional tools
rustup component add clippy rustfmt llvm-tools-preview
cargo install cargo-llvm-cov

# Build
cargo build

# Run tests
cargo test

# Check code quality
cargo clippy -- -D warnings
```

## Optional: HuggingFace CLI

For downloading models manually:

```bash
pip install huggingface-hub
```

## Next Steps

- [First Example](./first-example.md) - Create your first program
- [Quick Start](./quick-start.md) - Common usage patterns
- [User Guide](./guide/devices.md) - Detailed documentation

## Platform Support

| Platform | Support | Notes |
|----------|---------|-------|
| Apple Silicon (M1/M2/M3/M4) | ✅ Full | Primary target |
| Intel Mac | ❌ Not supported | No Metal backend |
| Linux | ❌ Not supported | No Metal on Linux |
| Windows | ❌ Not supported | Metal is macOS only |

metal-candle is specifically designed for Apple Silicon. For other platforms, consider:
- [Candle](https://github.com/huggingface/candle) - Cross-platform ML framework
- [burn](https://github.com/tracel-ai/burn) - Rust ML framework
- [PyTorch](https://pytorch.org) - Python ML framework

## License

metal-candle is licensed under Apache-2.0. See [LICENSE](https://github.com/GarthDB/metal-candle/blob/main/LICENSE).
