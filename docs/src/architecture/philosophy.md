# Design Philosophy

Core principles guiding metal-candle.

## Quality Over Speed

Production-quality code from day one:
- Zero warnings (pedantic clippy)
- ≥80% test coverage
- 100% API documentation
- Proper error handling (no `.unwrap()`)

## Explicit Over Implicit

Clear, readable code preferred:

```rust
// Good: Explicit
let tensor = Tensor::zeros((batch, seq), DType::F32, &device)?;

// Bad: Implicit device
let tensor = Tensor::zeros((batch, seq))?;
```

## Fail Fast

Use `Result` and `?` operator:

```rust
// Good: Explicit error handling
pub fn load(path: impl AsRef<Path>) -> Result<Model, ModelError> {
    // ...
}

// Bad: Hidden failures
pub fn load(path: impl AsRef<Path>) -> Model {
    // .unwrap() or .expect()
}
```

## Specialized, Not General

Focus on LoRA training excellence:
- Type-safe LoRA implementation with comprehensive testing
- Not trying to be general-purpose ML framework
- Deep rather than broad
- Production-quality over feature breadth

## Apple Silicon First

Designed specifically for Metal GPU:
- Not cross-platform compromise
- Leverage unified memory architecture
- Direct Metal integration

## Zero-Cost Abstractions

Rust's strengths:
- Compile-time optimization
- No runtime overhead
- Memory safety without GC

## Single Binary Deployment

No Python, no virtual environments:
- Pure Rust
- Static linking
- Simple deployment

## See Also

- [Performance](./performance.md)
- [Backend](./backend.md)
