# Code Style

Coding standards for metal-candle.

## Rust Style

Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

### Key Principles

1. **Explicit over implicit**
2. **Builder patterns** for complex configuration
3. **Sensible defaults**
4. **Clear error messages**

### Example

```rust
// Good: Builder pattern with defaults
let model = ModelLoader::new()
    .with_device(Device::metal(0)?)
    .with_dtype(DType::F16)
    .load("path/to/model")?;

// Good: Explicit error types
#[derive(Error, Debug)]
pub enum LoadError {
    #[error("file not found: {path}")]
    FileNotFound { path: PathBuf },
}
```

## Formatting

Use `rustfmt` with default settings:

```bash
cargo fmt
```

## Linting

Pedantic clippy:

```bash
cargo clippy -- -D warnings
```

## Documentation

Every public item needs:
- Summary (one sentence)
- Description (what, when, why)
- Examples (runnable)
- Errors section
- Panics section

## See Also

- [CONTRIBUTING.md](https://github.com/GarthDB/metal-candle/blob/main/CONTRIBUTING.md)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
