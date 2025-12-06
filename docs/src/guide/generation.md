# Text Generation

Generating text with transformer models using KV-cache and sampling strategies.

## Quick Start

```rust
use metal_candle::inference::{
    KVCache, KVCacheConfig, SamplingStrategy, sample_token
};

// 1. Setup KV-cache
let cache_config = KVCacheConfig {
    max_seq_len: 2048,
    num_layers: 24,
    num_heads: 14,
    head_dim: 64,
    batch_size: 1,
};
let mut cache = KVCache::new(cache_config, &device)?;

// 2. Generate tokens
let strategy = SamplingStrategy::TopP { p: 0.9 };

for position in 0..max_new_tokens {
    // Forward pass (uses cache automatically)
    let logits = model.forward_with_cache(&input_ids, position, &mut cache)?;
    
    // Sample next token
    let next_token = sample_token(&logits, &strategy)?;
    
    // Add to sequence
    input_ids.push(next_token);
}
```

## KV-Cache

The KV-cache dramatically speeds up generation by caching attention keys/values.

### Configuration

```rust
use metal_candle::inference::KVCacheConfig;

let config = KVCacheConfig {
    max_seq_len: 2048,      // Maximum sequence length
    num_layers: 24,          // Number of transformer layers
    num_heads: 14,           // Number of attention heads
    head_dim: 64,            // Dimension per head
    batch_size: 1,           // Batch size (usually 1 for generation)
};

let cache = KVCache::new(config, &device)?;
```

### Memory Usage

Formula: `layers × 2 × batch × heads × seq_len × head_dim × bytes`

Example (Qwen 0.5B, F16, 2048 tokens):
```
24 × 2 × 1 × 14 × 2048 × 64 × 2 = ~173 MB
```

### Cache Operations

```rust
// Create cache
let mut cache = KVCache::new(config, &device)?;

// Use in forward pass
let logits = model.forward_with_cache(&tokens, position, &mut cache)?;

// Clear cache (start new sequence)
cache.clear()?;

// Check cache state
println!("Cached positions: {}", cache.current_position());
```

### Performance

KV-cache overhead is <1% of total generation time:
- Cache update: **12 ns** per layer
- 24 layers: **337 ns** total
- Negligible compared to forward pass (~10ms)

## Sampling Strategies

Different strategies for selecting the next token.

### Greedy Sampling

Always pick the highest probability token:

```rust
use metal_candle::inference::SamplingStrategy;

let strategy = SamplingStrategy::Greedy;
let token = sample_token(&logits, &strategy)?;
```

**Use for:**
- Deterministic output
- Maximum likelihood generation
- Testing/debugging

### Temperature Sampling

Control randomness:

```rust
// Low temperature (more focused)
let strategy = SamplingStrategy::Temperature { temperature: 0.7 };

// High temperature (more random)
let strategy = SamplingStrategy::Temperature { temperature: 1.5 };

let token = sample_token(&logits, &strategy)?;
```

**Guidelines:**
- `temperature < 1.0`: More focused, less random
- `temperature = 1.0`: Standard sampling
- `temperature > 1.0`: More creative, more random

### Top-k Sampling

Sample from top-k most likely tokens:

```rust
let strategy = SamplingStrategy::TopK { k: 50 };
let token = sample_token(&logits, &strategy)?;
```

**Common values:**
- `k = 10`: Very focused
- `k = 50`: Balanced (recommended)
- `k = 100`: More diverse

### Top-p (Nucleus) Sampling

Sample from smallest set with cumulative probability >= p:

```rust
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits, &strategy)?;
```

**Common values:**
- `p = 0.9`: Recommended default
- `p = 0.95`: More diverse
- `p = 0.8`: More focused

### Combined Strategies

Combine temperature with top-p:

```rust
// 1. Apply temperature scaling
let logits_scaled = logits / temperature;

// 2. Apply top-p sampling
let strategy = SamplingStrategy::TopP { p: 0.9 };
let token = sample_token(&logits_scaled, &strategy)?;
```

## Complete Generation Example

```rust
use metal_candle::*;
use anyhow::Result;

fn generate_text(
    model: &Qwen,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    let device = model.device();
    
    // 1. Tokenize prompt
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    
    // 2. Setup KV-cache
    let cache_config = KVCacheConfig {
        max_seq_len: 2048,
        num_layers: 24,
        num_heads: 14,
        head_dim: 64,
        batch_size: 1,
    };
    let mut cache = KVCache::new(cache_config, device)?;
    
    // 3. Configure sampling
    let strategy = SamplingStrategy::TopP { p: 0.9 };
    
    // 4. Generate tokens
    for position in 0..max_tokens {
        // Forward pass
        let input_ids = Tensor::from_vec(
            tokens.clone(),
            (1, tokens.len()),
            device,
        )?;
        
        let logits = model.forward_with_cache(&input_ids, position, &mut cache)?;
        
        // Get last token logits
        let last_logits = logits.i((0, tokens.len() - 1))?;
        
        // Sample next token
        let next_token = sample_token(&last_logits, &strategy)?;
        
        // Check for EOS
        if next_token == tokenizer.token_to_id("<|endoftext|>") {
            break;
        }
        
        tokens.push(next_token);
    }
    
    // 5. Decode to text
    let generated = tokenizer.decode(&tokens, true)?;
    Ok(generated)
}
```

## Streaming Generation

For real-time output:

```rust
fn generate_streaming(
    model: &Qwen,
    tokenizer: &Tokenizer,
    prompt: &str,
    callback: impl Fn(&str),
) -> Result<()> {
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    let mut cache = KVCache::new(config, model.device())?;
    
    for position in 0..max_tokens {
        let logits = model.forward_with_cache(&tokens, position, &mut cache)?;
        let token = sample_token(&logits, &strategy)?;
        
        tokens.push(token);
        
        // Decode and stream current token
        let text = tokenizer.decode(&[token], false)?;
        callback(&text);
    }
    
    Ok(())
}

// Usage
generate_streaming(&model, &tokenizer, "Hello", |token_text| {
    print!("{}", token_text);
    std::io::stdout().flush().unwrap();
})?;
```

## Performance Tips

### 1. Use KV-Cache

```rust
// With cache (fast!)
let logits = model.forward_with_cache(&tokens, pos, &mut cache)?;

// Without cache (slow - recomputes everything!)
let logits = model.forward(&tokens)?;
```

**Speedup:** >10x for long sequences

### 2. Use CPU for Sampling

```rust
// Model forward on GPU
let logits_gpu = model.forward_with_cache(&tokens, pos, &mut cache)?;

// Sampling on CPU (more efficient for small ops)
let logits_cpu = logits_gpu.to_device(&Device::Cpu)?;
let token = sample_token(&logits_cpu, &strategy)?;
```

### 3. Batch Generation (v1.1+)

Generate multiple sequences in parallel:

```rust
// Future feature
let config = KVCacheConfig {
    batch_size: 4,  // Generate 4 sequences at once
    ..config
};
```

## Stopping Criteria

### End-of-Sequence Token

```rust
// Stop when EOS token is generated
if next_token == eos_token_id {
    break;
}
```

### Maximum Length

```rust
// Stop after max_tokens
for position in 0..max_tokens {
    // ...generation...
}
```

### Custom Stopping

```rust
// Stop when specific condition met
if generated_text.ends_with("END") {
    break;
}
```

## Troubleshooting

### "Cache position out of bounds"

Increase `max_seq_len` in cache config:
```rust
let config = KVCacheConfig {
    max_seq_len: 4096,  // Increased from 2048
    ..config
};
```

### Repetitive output

Try different sampling strategies:
```rust
// Add temperature
let strategy = SamplingStrategy::Temperature { temperature: 0.8 };

// Or use top-p
let strategy = SamplingStrategy::TopP { p: 0.95 };
```

### Nonsensical output

Lower temperature or use top-k:
```rust
let strategy = SamplingStrategy::TopK { k: 50 };
```

## See Also

- [KV-Cache Benchmarks](../testing/benchmarks.md#kv-cache-performance)
- [Sampling Performance](../testing/benchmarks.md#sampling-overhead)
- [Model Loading](./models.md)
