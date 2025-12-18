# GitHub Issue Draft: Implement ApplyAdapter Trait for Hot-Swapping

**Title**: Implement `ApplyAdapter` trait for Qwen model (v1.3.1)

**Labels**: `enhancement`, `v1.3.1`, `LoRA`, `priority:high`

---

## Summary

Implement the `ApplyAdapter` trait for the `Qwen` model to enable true hot-swapping of LoRA adapters without manual model integration. This completes the adapter management feature introduced in v1.3.0.

## Background

In v1.3.0 (#48), we introduced:
- ✅ `AdapterRegistry` for managing multiple LoRA adapters
- ✅ `ApplyAdapter` trait definition
- ⏳ Trait implementation (deferred to v1.3.1)

Currently, the `AdapterRegistry` manages adapter storage and activation state, but users must manually integrate adapters with their model's forward pass. This issue tracks the implementation of automatic adapter integration.

## Goals

### 1. Implement `ApplyAdapter` for `Qwen` Model

Implement the three methods defined in `src/training/apply_adapter.rs`:

```rust
pub trait ApplyAdapter {
    fn apply_adapter(&mut self, adapter: &LoRAAdapter) -> Result<()>;
    fn remove_adapter(&mut self) -> Result<()>;
    fn has_adapter(&self) -> bool;
}
```

### 2. Refactor `Qwen` Model Structure

Current structure needs modification:

**Before**:
```rust
pub struct Qwen {
    embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
}
```

**After**:
```rust
pub struct Qwen {
    embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    lm_head: Linear,
    adapter: Option<LoRAAdapter>,  // NEW: Optional adapter storage
}
```

### 3. Update Forward Pass

Modify `Qwen::forward()` to conditionally apply adapter when present:

```rust
impl Qwen {
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden_states = self.embedding.forward(input_ids)?;
        
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, attention_mask)?;
            
            // Apply adapter if present
            if let Some(adapter) = &self.adapter {
                hidden_states = adapter.forward(&hidden_states, block_idx)?;
            }
        }
        
        // ... rest of forward pass
    }
}
```

## Technical Requirements

### 1. Model Integration
- [ ] Add `adapter: Option<LoRAAdapter>` field to `Qwen` struct
- [ ] Update `Qwen::new()` to initialize `adapter` as `None`
- [ ] Modify `forward()` to apply adapter when present
- [ ] Ensure adapter is applied at correct layer positions

### 2. Trait Implementation
- [ ] Implement `apply_adapter()`: Store adapter reference and enable in forward pass
- [ ] Implement `remove_adapter()`: Clear adapter and revert to base model
- [ ] Implement `has_adapter()`: Return whether adapter is currently applied
- [ ] Handle adapter dimension mismatches with clear error messages

### 3. Performance
- [ ] Adapter application: <100ms (target: <50ms)
- [ ] Adapter removal: <50ms
- [ ] Forward pass overhead with adapter: <5% vs without adapter
- [ ] Memory overhead: Only adapter weights (no base model duplication)

### 4. Testing
- [ ] Unit tests for `ApplyAdapter` implementation
- [ ] Integration test: `Registry` → `activate()` → `apply_adapter()` → `generate()`
- [ ] Test adapter switching: Apply adapter A → generate → switch to B → generate
- [ ] Test error handling: Incompatible adapter dimensions
- [ ] Benchmark adapter switching latency
- [ ] Verify memory efficiency (no model duplication)

### 5. Examples
- [ ] Update `adapter_swap_demo.rs` to demonstrate full hot-swapping
- [ ] Add example showing adapter switching during inference
- [ ] Document memory savings and performance characteristics

### 6. Documentation
- [ ] Update `ApplyAdapter` trait docs with implementation notes
- [ ] Document `Qwen` adapter integration in module docs
- [ ] Update README with complete hot-swapping workflow
- [ ] Update CHANGELOG for v1.3.1
- [ ] Add migration guide from v1.3.0 manual integration

## Implementation Plan

### Phase 1: Core Implementation (Week 1)
1. Add `adapter` field to `Qwen` struct
2. Implement `ApplyAdapter` trait for `Qwen`
3. Update forward pass to conditionally apply adapter
4. Add unit tests for each trait method

### Phase 2: Integration & Testing (Week 1-2)
5. Create end-to-end integration test with real model
6. Test adapter switching scenarios
7. Benchmark performance (application, removal, forward pass)
8. Verify memory efficiency

### Phase 3: Documentation & Examples (Week 2)
9. Update examples to show full hot-swapping
10. Write comprehensive documentation
11. Update CHANGELOG and README
12. Review and test migration from v1.3.0

## Success Criteria

- [ ] All `ApplyAdapter` methods implemented for `Qwen`
- [ ] Adapter switching <100ms latency
- [ ] Forward pass overhead <5% with adapter
- [ ] Zero base model memory duplication
- [ ] All tests passing (including new integration tests)
- [ ] Examples demonstrate full workflow
- [ ] Documentation complete with migration guide

## Related Issues

- #48 - v1.3.0 PR that introduced `AdapterRegistry` and `ApplyAdapter` trait
- (Add any related issues about LoRA training, inference, etc.)

## Notes

### Design Considerations

1. **Adapter Storage**: Store `Option<LoRAAdapter>` vs `Option<&LoRAAdapter>`?
   - Recommendation: Use owned `LoRAAdapter` to avoid lifetime issues
   - Consider `Arc<LoRAAdapter>` if sharing needed across threads

2. **Layer Matching**: How to match adapter layers to model blocks?
   - Current `LoRAAdapter` stores layers by index
   - Ensure adapter `num_layers` matches model `blocks.len()`
   - Add validation in `apply_adapter()` with clear error message

3. **Target Modules**: Handle adapters with different target modules?
   - Adapter may target Q/K/V/O projections
   - Model blocks need flexible application logic
   - Document supported target module combinations

4. **Thread Safety**: Should `apply_adapter()` take `&mut self` or use interior mutability?
   - Current signature uses `&mut self` (simpler)
   - Consider `Arc<RwLock<Option<LoRAAdapter>>>` for concurrent inference
   - Defer concurrency to v2.0 if not needed for v1.3.1

### Future Enhancements (v1.4+)

- Support multiple adapters simultaneously (mixture of adapters)
- Adapter interpolation/merging
- Dynamic adapter scaling (adjust alpha at runtime)
- Adapter-specific target modules per block
- Thread-safe concurrent adapter switching

## References

- LoRA Paper: https://arxiv.org/abs/2106.09685
- Current `ApplyAdapter` trait: `src/training/apply_adapter.rs`
- Current `AdapterRegistry`: `src/training/adapter_registry.rs`
- Current `Qwen` model: `src/models/qwen.rs`

---

**Assignee**: TBD  
**Milestone**: v1.3.1  
**Priority**: High (completes v1.3.0 feature)  
**Estimated Effort**: 1-2 weeks


