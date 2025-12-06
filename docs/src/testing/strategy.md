# Testing Strategy

How metal-candle ensures quality.

## Test Coverage

**Target**: ≥80% line coverage  
**Current**: 84.69% ✅

**By Module**:
- Public APIs: 100%
- Core algorithms: 100%
- Utilities: ≥80%

## Test Types

### Unit Tests

In same file as implementation:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature() {
        // Test code
    }
}
```

### Integration Tests

In `tests/` directory:

```rust
// tests/integration/my_feature.rs
use metal_candle::*;

#[test]
fn test_end_to_end() {
    // Integration test
}
```

### Doc Tests

In documentation:

```rust
/// # Examples
///
/// ```
/// use metal_candle::*;
/// // Example that runs as test
/// ```
```

## Running Tests

```bash
cargo test  # All tests
cargo test training  # Specific module
cargo test --doc  # Doc tests only
```

## Coverage

```bash
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html
```

## CI Testing

Tests run on:
- Every PR
- Every push to main
- Apple Silicon runners (macos-14)

## See Also

- [Code Coverage](./coverage.md)
- [CONTRIBUTING.md](https://github.com/GarthDB/metal-candle/blob/main/CONTRIBUTING.md)
