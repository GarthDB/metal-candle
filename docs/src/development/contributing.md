# Contributing Guide

Guidelines for contributing to metal-candle.

## Quick Links

For complete contributing guidelines, see:

**[CONTRIBUTING.md](https://github.com/GarthDB/metal-candle/blob/main/CONTRIBUTING.md)**

## Quick Start

```bash
# Clone
git clone https://github.com/GarthDB/metal-candle
cd metal-candle

# Install tools
rustup component add clippy rustfmt llvm-tools-preview
cargo install cargo-llvm-cov

# Build & Test
cargo build
cargo test
cargo clippy -- -D warnings
```

## Code Quality Standards

### Zero Warnings Policy

```bash
cargo clippy -- -D warnings  # Must pass
```

**Linting Level**: Pedantic (strict)

### Code Coverage

**Target**: ≥80% coverage

```bash
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html
```

### Formatting

```bash
cargo fmt  # Apply formatting
cargo fmt --check  # Verify
```

## Testing

### Write Tests

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

**Coverage Requirements**:
- Public APIs: 100%
- Core algorithms: 100%
- Utilities: ≥80%

### Run Tests

```bash
cargo test  # All tests
cargo test training  # Specific module
cargo test -- --nocapture  # Show output
```

## Documentation

### Document Everything

```rust
/// Brief description.
///
/// Longer description explaining what, when, why.
///
/// # Examples
///
/// ```no_run
/// // Example code
/// ```
///
/// # Errors
///
/// Describe error conditions.
pub fn my_function() -> Result<()> {
    // ...
}
```

### Build Docs

```bash
cargo doc --all-features --no-deps --open
```

## Pull Request Process

### Before Submitting

- [ ] `cargo clippy -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] `cargo fmt` applied
- [ ] Tests added for new code
- [ ] Public APIs documented
- [ ] No `.unwrap()` in library code

### PR Template

```markdown
## Description
What changed and why

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Testing
- [ ] Tests pass
- [ ] New tests added

## Checklist
- [ ] Code quality checks pass
- [ ] Documentation updated
```

## Code Style

### Error Handling

**Use `thiserror`** for library code:

```rust
#[derive(Error, Debug)]
pub enum MyError {
    #[error("something failed: {0}")]
    Failed(String),
}
```

**Never use** `.unwrap()` or `.expect()` in library code.

### Naming

- Types: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Be descriptive, not clever

### Comments

```rust
// Explain WHY, not WHAT
// Good: "Pre-transpose to reduce kernel launches"
// Bad: "Transpose the matrix"
```

## Getting Help

- [GitHub Discussions](https://github.com/GarthDB/metal-candle/discussions)
- [Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct)

Thank you for contributing! 🎉
