# MPS Unsafe Code Strategy

**Date**: December 10, 2024  
**Status**: 🔒 Security Planning  
**Goal**: Minimize unsafe code surface area while enabling MPS FFI

---

## Problem Statement

Metal Performance Shaders (MPS) integration requires Objective-C FFI via the `objc` crate, which uses `unsafe` code for:
- `msg_send!` macro for Objective-C method calls
- Raw pointer manipulation for buffer sharing
- Manual memory management (retain/release)

We had `unsafe_code = "forbid"` at the crate level for safety, but temporarily disabled it to compile MPS. **This is too permissive.**

---

## Security Concerns

### Current Risk (Crate-Wide Unsafe Allowed)
- ❌ Any module can now use `unsafe` without restriction
- ❌ Loses compile-time safety guarantees across entire codebase
- ❌ Could introduce memory safety bugs in non-FFI code
- ❌ Violates our "production quality" standard

### MPS-Specific Risks
Even with proper isolation, MPS FFI has inherent risks:
- **Memory leaks**: Incorrect retain/release balance
- **Use-after-free**: Dropped Rust wrapper, dangling Objective-C object
- **Null pointer dereference**: MPS returns null on failure
- **Type confusion**: Incorrect Objective-C type casting
- **Thread safety**: Objective-C objects may not be Send/Sync

---

## Immediate Fix (Day 3)

### Use Crate-Level Deny (Not Forbid)

**Important**: `forbid` cannot be overridden by module-level `allow`, so we use `deny`.

**Cargo.toml**:
```toml
[lints.rust]
unsafe_code = "deny"  # Errors by default, but MPS can opt-in with #![allow]
```

**Difference**:
- `forbid`: Cannot be overridden (too strict for our needs)
- `deny`: Errors unless explicitly allowed (perfect for controlled exceptions)
- `warn`: Only warns (too permissive)

### Module-Level Exceptions

**src/backend/mps/mod.rs**:
```rust
//! MPS FFI requires unsafe for Objective-C interop.
//! All unsafe code is isolated to this module and audited for safety.

#![cfg(feature = "mps")]
#![allow(unsafe_code)]  // EXCEPTION: Required for MPS FFI only
```

**Result**: Unsafe code allowed ONLY in `src/backend/mps/` modules, forbidden everywhere else.

---

## Best Practices for MPS Unsafe Code

### 1. RAII Memory Management

**Good** (current implementation):
```rust
pub struct MPSMatrix {
    inner: *mut Object,
}

impl Drop for MPSMatrix {
    fn drop(&mut self) {
        unsafe {
            let _: () = msg_send![self.inner, release];
        }
    }
}
```

**Why Safe**: Rust's ownership system ensures `release` is called exactly once.

### 2. Null Pointer Checks

**Good** (current implementation):
```rust
let descriptor: *mut Object = msg_send![...];
if descriptor.is_null() {
    return Err(...);  // Safe error path
}
let _: () = msg_send![descriptor, retain];  // Only retain if non-null
```

**Why Safe**: Validates MPS return values before use.

### 3. Minimal Unsafe Surface

**Bad**:
```rust
pub fn raw_mps_call(&self) -> *mut Object {
    unsafe { msg_send![self.inner, someMethod] }
}
```

**Good**:
```rust
pub fn safe_mps_call(&self) -> Result<SafeWrapper> {
    unsafe {
        let raw: *mut Object = msg_send![self.inner, someMethod];
        if raw.is_null() {
            return Err(...);
        }
        let _: () = msg_send![raw, retain];
        Ok(SafeWrapper { inner: raw })
    }
}
```

**Why**: Public API is 100% safe, unsafe is internal implementation detail.

### 4. Document Safety Invariants

**Good**:
```rust
impl MPSMatrix {
    /// Create a new MPSMatrix.
    ///
    /// # Safety Invariants
    ///
    /// - `inner` is retained on creation
    /// - `inner` is released on drop
    /// - `inner` is never null
    /// - `inner` is never accessed after drop
    pub fn new(...) -> Result<Self> {
        // ...
    }
}
```

---

## Long-Term Strategies

### Option 1: Higher-Level Bindings (Preferred)

**Timeline**: Post-v2.0 (if available)

Monitor for safer MPS bindings:
- Wait for `metal-rs` to add MPS support
- Contribute MPS bindings to `metal-rs` ourselves
- Use `cxx` bridge for C++/Objective-C++ wrapper

**Benefit**: Eliminates manual `msg_send!` and retain/release.

### Option 2: Isolated FFI Layer

**Timeline**: Week 12 (refinement)

Create thin, audited FFI layer:
```
src/backend/mps/
  ├── mod.rs           # Public safe API
  ├── ffi/             # Unsafe FFI layer (isolated)
  │   ├── mod.rs       # #![allow(unsafe_code)]
  │   ├── descriptor.rs
  │   ├── matrix.rs
  │   └── kernel.rs
  ├── matmul.rs        # Safe wrappers (no unsafe)
  ├── softmax.rs       # Safe wrappers (no unsafe)
  └── rms_norm.rs      # Safe wrappers (no unsafe)
```

**Benefit**: Unsafe code in one small submodule, rest is safe.

### Option 3: Formal Safety Audit

**Timeline**: Pre-v2.0 release

Before production use:
1. **Manual audit**: Review every `unsafe` block
2. **MIRI testing**: Run under Rust's MIRI for undefined behavior detection
3. **Instruments profiling**: Verify no memory leaks (we'll do this in Week 13)
4. **Fuzz testing**: Random inputs to catch edge cases

**Deliverable**: `MPS_SAFETY_AUDIT.md` documenting all unsafe code and safety proofs.

---

## Comparison to Existing Unsafe Code

We **already have unsafe code** in custom Metal kernels:

**src/backend/custom_ops.rs** (line 168):
```rust
fn get_or_compile_pipeline(&self, device: &metal::DeviceRef) -> Result<...> {
    // ...
    unsafe {
        let owned_device = device.to_owned();  // Unsafe pointer conversion
    }
    // ...
}
```

**Why MPS is Similar**:
- Both require Metal FFI
- Both need buffer pointer sharing
- Both use `metal` crate (which uses unsafe internally)

**Why MPS is Different**:
- More `unsafe` blocks (Objective-C calls)
- More complex memory management (retain/release)
- External dependency on Apple's MPS framework

---

## Testing Strategy for Unsafe Code

### 1. Leak Detection (Week 13)

```bash
# Run with Instruments to detect leaks
cargo instruments -t Leaks --release --features mps --test mps_tests
```

**Expected**: Zero leaks after 1000 iterations.

### 2. Thread Safety (Week 13)

```rust
#[test]
fn test_mps_thread_safety() {
    // Ensure MPSMatrix is Send/Sync if claimed
    let matrix = create_mps_matrix();
    std::thread::spawn(move || {
        drop(matrix);  // Should not cause issues
    }).join().unwrap();
}
```

### 3. Property-Based Testing (Week 12)

```rust
proptest! {
    #[test]
    fn mps_never_panics(
        rows in 1usize..1000,
        cols in 1usize..1000,
    ) {
        // MPS should handle any valid dimensions without panicking
        let result = create_mps_descriptor(rows, cols);
        assert!(result.is_ok() || result.is_err());  // Never panics
    }
}
```

---

## Mitigation Checklist

### Immediate (Day 3)
- [x] ~~Disable crate-level `unsafe_code` forbid~~ (DONE, but wrong)
- [ ] Re-enable crate-level `unsafe_code = "forbid"`
- [ ] Add module-level `#![allow(unsafe_code)]` only to MPS
- [ ] Verify non-MPS code cannot use unsafe

### Short-Term (Week 10-12)
- [ ] Document every `unsafe` block with safety proof
- [ ] Isolate unsafe to `mps/ffi/` submodule
- [ ] Add `SAFETY:` comments to all unsafe code
- [ ] Test with Instruments (leak detection)

### Long-Term (Post-v2.0)
- [ ] Monitor for safer MPS bindings
- [ ] Consider contributing to `metal-rs`
- [ ] Formal safety audit before production use
- [ ] Explore `cxx` bridge for Objective-C++

---

## Code Review Guidelines

Before merging any MPS code:

### ✅ Required
1. Every `unsafe` block has `// SAFETY:` comment explaining invariants
2. All public APIs are 100% safe (no unsafe in signatures)
3. RAII wrappers for all Objective-C objects
4. Null checks on all MPS return values
5. No raw pointers in public API

### 🔍 Review Checklist
- [ ] Retain/release balanced?
- [ ] Null pointers handled?
- [ ] Type casts correct?
- [ ] Thread safety considered?
- [ ] Drop implementation correct?

---

## Recommendation

### ✅ Short-Term (This Week)
Re-enable crate-level `forbid`, allow only in MPS module:

```toml
# Cargo.toml
[lints.rust]
unsafe_code = "forbid"
```

```rust
// src/backend/mps/mod.rs
#![allow(unsafe_code)]  // EXCEPTION: MPS FFI only
```

### ✅ Medium-Term (Week 12)
Isolate unsafe to `mps/ffi/` submodule, rest of MPS uses safe wrappers.

### ✅ Long-Term (v2.1+)
Eliminate unsafe entirely by:
1. Contributing to `metal-rs` for safe MPS bindings, OR
2. Using higher-level framework if available

---

## Conclusion

**Current Risk**: 🟡 Medium (crate-wide unsafe allowed)  
**After Immediate Fix**: 🟢 Low (isolated to MPS module only)  
**After Full Implementation**: 🟢 Acceptable (well-audited, tested, isolated)

**Action Item**: Re-enable `unsafe_code = "forbid"` at crate level, allow only in MPS module.

This maintains our "production quality" standard while enabling the performance benefits of MPS.

---

**Status**: Strategy defined, implementation in progress  
**Owner**: MPS integration team  
**Review**: Required before v2.0 release

