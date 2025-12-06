# Error Handling

Error handling architecture in metal-candle.

## Error Types

Using `thiserror` for structured errors:

### ModelError

```rust
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("model file not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("invalid model format: {reason}")]
    InvalidFormat { reason: String },
    
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
```

### TrainingError

For training operations:
- InvalidConfiguration
- GradientError
- CheckpointError

### InferenceError

For generation:
- CacheError
- SamplingError

## Usage

Library code:

```rust
pub fn load(path: impl AsRef<Path>) -> Result<Model, ModelError> {
    // Explicit error handling
    std::fs::read(path.as_ref())
        .map_err(|e| ModelError::Io(e))?;
    // ...
}
```

Application code can use `anyhow`:

```rust
use anyhow::Result;

fn main() -> Result<()> {
    let model = ModelLoader::new().load("model.safetensors")?;
    Ok(())
}
```

## Principles

### Never `.unwrap()`

In library code:

```rust
// Bad
let value = option.unwrap();

// Good
let value = option.ok_or(ModelError::MissingValue)?;
```

### Actionable Error Messages

```rust
// Good: Tells user what to do
#[error("model file not found: {path}. Ensure the file exists and path is correct.")]

// Bad: Vague
#[error("file error")]
```

### Convert Errors at Boundaries

```rust
impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}
```

## See Also

- [API Documentation](../reference/api.md)
- [Troubleshooting](../reference/troubleshooting.md)
