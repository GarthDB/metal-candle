# Your First Example

Let's create your first program using metal-candle! We'll load a model and generate embeddings.

## Prerequisites

- Rust 1.75+ installed
- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 12.0+

## Create a New Project

```bash
cargo new my-metal-app
cd my-metal-app
```

## Add metal-candle Dependency

Edit `Cargo.toml`:

```toml
[dependencies]
metal-candle = "1.0"
anyhow = "1.0"
```

## Write Your First Program

Replace `src/main.rs` with:

```rust
use metal_candle::embeddings::{EmbeddingModel, EmbeddingModelType};
use candle_core::Device;
use anyhow::Result;

fn main() -> Result<()> {
    println!("🚀 metal-candle First Example\n");
    
    // 1. Setup device (CPU for this simple example)
    let device = Device::Cpu;
    println!("📱 Device: {:?}", device);
    
    // 2. Load an embedding model (auto-downloads from HuggingFace)
    println!("\n📥 Loading E5-small-v2 model...");
    let model = EmbeddingModel::from_pretrained(
        EmbeddingModelType::E5SmallV2,
        device,
    )?;
    println!("✅ Model loaded! Dimension: {}", model.dimension());
    
    // 3. Create some text to encode
    let texts = vec![
        "Rust is a systems programming language",
        "Python is great for scripting",
        "Machine learning on Apple Silicon is fast",
    ];
    
    // 4. Generate embeddings
    println!("\n🔄 Generating embeddings...");
    let embeddings = model.encode(&texts)?;
    println!("✅ Generated embeddings: {:?}", embeddings.shape());
    
    // 5. Calculate similarity between first two texts
    let vecs = embeddings.to_vec2::<f32>()?;
    let similarity: f32 = vecs[0]
        .iter()
        .zip(&vecs[1])
        .map(|(a, b)| a * b)
        .sum();
    
    println!("\n📊 Similarity between first two texts: {:.4}", similarity);
    println!("\n✨ Success!");
    
    Ok(())
}
```

## Run It!

```bash
cargo run
```

**Output:**
```
🚀 metal-candle First Example

📱 Device: Cpu

📥 Loading E5-small-v2 model...
✅ Model loaded! Dimension: 384

🔄 Generating embeddings...
✅ Generated embeddings: [3, 384]

📊 Similarity between first two texts: 0.7234

✨ Success!
```

## What Just Happened?

1. **Device Setup**: We initialized a CPU device (Metal GPU also available)
2. **Model Loading**: Downloaded and loaded E5-small-v2 from HuggingFace
3. **Embedding Generation**: Converted text into 384-dimensional vectors
4. **Similarity Calculation**: Computed cosine similarity between embeddings

## Next Steps

Explore more examples:

- **[Device Management](./guide/devices.md)** - Using Metal GPU
- **[Model Loading](./guide/models.md)** - Loading transformer models
- **[LoRA Training](./guide/lora.md)** - Fine-tuning models
- **[Text Generation](./guide/generation.md)** - Generating text

## More Examples

Check out the repository's `examples/` directory:

```bash
# Clone the repository
git clone https://github.com/GarthDB/metal-candle
cd metal-candle

# Run examples
cargo run --example embeddings_demo --features embeddings
cargo run --example load_model
cargo run --example inference_demo
```

## Troubleshooting

**Model download fails?**
- Check internet connection
- Models cache in `~/.cache/ferris/models/`

**Build errors?**
- Ensure Rust 1.75+ (`rustc --version`)
- Ensure you're on Apple Silicon Mac

**Need help?**
- [GitHub Issues](https://github.com/GarthDB/metal-candle/issues)
- [Troubleshooting Guide](./reference/troubleshooting.md)
