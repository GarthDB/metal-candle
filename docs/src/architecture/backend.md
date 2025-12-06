# Backend Abstraction

Device and Metal backend architecture.

## Device Abstraction

Simple, explicit device management:

```rust
pub enum DeviceType {
    Metal,
    Cpu,
}

pub struct Device {
    // Candle device wrapper
}

impl Device {
    pub fn new_metal(index: usize) -> Result<Self>;
    pub fn new_cpu() -> Self;
    pub fn new_with_fallback(index: usize) -> Self;
}
```

## Metal Integration

Via Candle's Metal backend:
- Uses Apple's Metal Performance Shaders
- Unified memory architecture
- Automatic synchronization

## Tensor Operations

Wrapper around Candle tensors:
- Type safety
- Device tracking
- Error handling

## Design Decisions

### Why Candle?

- Mature Metal backend
- Active development
- Rust-native
- Good performance

### Why Not Custom Metal?

- Candle performance sufficient (110% of MLX)
- Maintenance burden too high
- Can optimize specific ops if needed

## See Also

- [Performance](./performance.md)
- [Philosophy](./philosophy.md)
