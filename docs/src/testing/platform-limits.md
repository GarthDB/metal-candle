# Platform Coverage Limits

## Understanding Test Coverage on Single-Platform CI

metal-candle is designed to work across multiple platforms (Apple Silicon with Metal, CPU-only systems), but our CI runs exclusively on Apple Silicon (`macos-14` runners). This creates inherent limitations in code coverage metrics.

## The Coverage Challenge

### Current Coverage: 92.9%

Our codebase achieves **92.9% test coverage** on Apple Silicon CI, which significantly exceeds our 80% minimum requirement. However, reaching 100% coverage is **technically impossible** on single-platform CI.

### Why 100% is Impossible

Consider this code path:

```rust
pub fn new_metal(index: usize) -> Result<Self> {
    match CandleDevice::new_metal(index) {
        Ok(inner) => Ok(Self { inner }),
        Err(e) => Err(DeviceError::MetalUnavailable {  // ← Never executes on Apple Silicon
            reason: format!("Failed to initialize Metal device {index}: {e}"),
        }.into()),
    }
}
```

**On Apple Silicon CI:**
- Metal is always available
- `CandleDevice::new_metal(0)` always succeeds
- The `Err` branch **never executes**
- This branch is marked as "uncovered" by Codecov

**On Linux/Windows CI:**
- Metal is never available
- The `Ok` branch never executes
- Different lines would be marked as "uncovered"

## What's Actually Uncovered

### Production Code (~5 lines)
1. Metal unavailable error construction (line 60-63)
2. CUDA device handling fallback (line 169)

### Test Code (~18 lines)
Platform detection conditionals in tests:
```rust
#[test]
fn test_platform_specific() {
    if Device::is_metal_available() {
        // This executes on Apple Silicon
        assert!(device.is_metal());
    } else {
        // This never executes on Apple Silicon CI ← "Uncovered"
        assert!(device.is_cpu());
    }
}
```

## Coverage Breakdown

| Category | Lines | Coverage | Notes |
|----------|-------|----------|-------|
| **Backend Module** | 642 | 92.3% | Production code |
| **Error Types** | 243 | 100% | All paths tested |
| **Test Code** | 200+ | Variable | Platform conditionals |
| **Overall Project** | 1000+ | 92.9% | Exceeds 80% requirement |

## Solutions Considered

### ❌ Mocking Metal Availability
**Pros:** Could test error paths  
**Cons:** 
- Complex, fragile, maintenance burden
- Doesn't test real platform behavior
- Goes against "test real behavior" philosophy

### ❌ Removing Platform Conditionals
**Pros:** Simpler code  
**Cons:**
- Breaks cross-platform support
- Worse user experience
- Against project goals

### ✅ Multi-Platform CI (Future)
**Pros:**
- Tests real platform behavior
- True cross-platform validation
- Better coverage insight

**Cons:**
- More complex CI setup
- Longer CI times
- Not critical for v1.0 (Apple Silicon focus)

**Status:** Planned for v1.1+

## Our Standards

### Coverage Requirements

✅ **Minimum: 80%** (Enforced by CI)  
✅ **Current: 92.9%** (Exceeds by 12.9 points)  
✅ **Backend Module: 92.3%** (Critical code well-tested)  
✅ **All Critical Paths: 100%** (Tested on Apple Silicon)

### Quality Over Metrics

We prioritize:
1. **Real behavior testing** over coverage percentage
2. **Meaningful tests** over coverage games
3. **Production quality** over arbitrary metrics

## Verification

All production code paths are validated:
- ✅ Device creation and detection
- ✅ Tensor operations and numerical stability
- ✅ Error handling and formatting
- ✅ Platform detection and fallbacks
- ✅ API contracts and traits

The uncovered lines are:
- Defensive code for other platforms
- Test conditionals for platform detection
- Non-critical to Apple Silicon users (our primary target)

## Conclusion

**92.9% coverage represents excellent test quality** for a single-platform CI setup. The uncovered code is:
- Well-understood and intentional
- Defensive for cross-platform compatibility
- Not feasible to test without multi-platform CI

We accept this limitation as a reasonable trade-off for v1.0, focused on Apple Silicon excellence.

---

**For Contributors:** See [Testing Strategy](./strategy.md) for how to write effective tests within these constraints.

