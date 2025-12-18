# Release Process for metal-candle

This document outlines the complete release process for metal-candle, ensuring production-quality releases with validated performance claims.

## Release Phases

### Phase 1: Pre-Release Validation (1-2 days)
### Phase 2: Official Benchmarks (1 day)
### Phase 3: Documentation Updates (2-3 hours)
### Phase 4: Release Execution (1 hour)
### Phase 5: Post-Release (1 hour)

---

## Phase 1: Pre-Release Validation

### 1.1 Code Quality Checks

Run all quality checks locally:

```bash
# Zero clippy warnings (pedantic mode)
cargo clippy --all-features --workspace -- -D warnings

# All tests passing
cargo test --all-features --workspace

# Code coverage ≥80%
cargo llvm-cov --all-features --workspace --summary-only

# Format check
cargo fmt --all -- --check

# Documentation builds
cargo doc --all-features --workspace --no-deps
```

**Acceptance Criteria:**
- ✅ Zero clippy warnings
- ✅ All tests passing
- ✅ Coverage ≥80%
- ✅ Docs build without warnings

### 1.2 Integration Testing

Test real-world scenarios:

```bash
# Run all examples
cargo run --example train_lora --features lora
cargo run --example inference_kv_cache --features inference
cargo run --example generate_text
cargo run --example embeddings --features embeddings

# Test on both CPU and Metal
METAL_DEVICE_WRAPPER_TYPE=1 cargo test --all-features
```

**Acceptance Criteria:**
- ✅ All examples run successfully
- ✅ No panics or unwrap failures
- ✅ Metal and CPU backends both work

### 1.3 Dependency Audit

```bash
# Check for security vulnerabilities
cargo deny check advisories

# Check for unmaintained dependencies
cargo deny check bans

# Update dependencies (if needed)
cargo update
cargo test --all-features  # Verify still works
```

**Acceptance Criteria:**
- ✅ No critical security issues
- ✅ Documented any unmaintained dependencies

---

## Phase 2: Official Benchmarks

**⚠️ CRITICAL:** Performance claims require controlled hardware benchmarks.

### 2.1 Prepare Benchmark Environment

```bash
# Close all non-essential apps
killall Chrome Safari Slack Discord "Google Chrome" || true

# Verify on power (not battery)
pmset -g batt

# Wait for system to cool down
# Check Activity Monitor: CPU should be <5% idle

# Verify Metal is available
cargo test backend::device::tests::test_metal_device_creation_on_macos
```

**Environment Checklist:**
- [ ] Running on Apple Silicon (M1/M2/M3/M4)
- [ ] Connected to power (not battery)
- [ ] System idle (<5% CPU usage)
- [ ] No high-CPU processes running
- [ ] Room temperature stable
- [ ] Metal GPU available

### 2.2 Run Official Benchmarks

```bash
# Run official benchmark suite (5 runs each)
./scripts/run_official_benchmarks.sh

# For quick validation (testing the script):
./scripts/run_official_benchmarks.sh --quick

# Skip MLX comparison if MLX not installed:
./scripts/run_official_benchmarks.sh --no-mlx
```

This will:
1. Verify environment is suitable
2. Run each benchmark 5 times with cooldown periods
3. Save results to `benchmark_results/YYYYMMDD_HHMMSS/`
4. Generate environment snapshot

**Time Required:** ~30-60 minutes (depending on benchmarks)

### 2.3 Analyze Benchmark Results

```bash
cd benchmark_results/YYYYMMDD_HHMMSS/

# Review all runs for consistency
# Calculate median values (most robust against outliers)
# Identify any anomalies (thermal throttling, etc.)
```

**Analysis Checklist:**
- [ ] All runs completed successfully
- [ ] Variance between runs <5% (good) or <10% (acceptable)
- [ ] No obvious outliers (thermal throttling, etc.)
- [ ] Performance meets or exceeds targets

**Performance Targets (v1.2.0+):**
- LoRA forward: ≥90% of MLX performance (1.5-3x faster)
- Fused Softmax: ≥1.5x faster than Candle baseline
- Embeddings: ≥20x faster than MLX
- Inference: ≥95% of MLX performance

### 2.4 Update Benchmark Documentation

```bash
# Update BENCHMARKS.md with median results
vim BENCHMARKS.md

# Add detailed results to git
git add benchmark_results/YYYYMMDD_HHMMSS/
git commit -m "docs: official benchmark results for v1.X.Y"
```

**Update in BENCHMARKS.md:**
```markdown
## v1.X.Y Benchmark Results (Dec 2025)

**Hardware:** M1 Max, 32GB RAM
**OS:** macOS 14.5
**Date:** December 10, 2025

| Operation | Time (µs) | vs MLX | vs Baseline |
|-----------|-----------|--------|-------------|
| LoRA Forward (512x512, r=8) | 42.3 | 2.1x faster | - |
| Fused Softmax (4x16x1024) | 15.8 | 3.2x faster | 1.9x faster |
...
```

---

## Phase 3: Documentation Updates

### 3.1 Update CHANGELOG.md

Move `[Unreleased]` section to new version:

```markdown
## [Unreleased]

(empty)

## [1.X.Y] - YYYY-MM-DD

### Highlights
- (Copy from Unreleased)

### Performance
- Validated on M1 Max: [specific claims from benchmarks]
```

### 3.2 Update Version Numbers

```bash
# Update Cargo.toml version
vim Cargo.toml
# Change: version = "1.X.Y"

# Update README.md installation instructions
vim README.md
# Change: metal-candle = "1.X"

# Update any hardcoded versions in docs
rg "1\.\d+\.\d+" docs/ examples/
```

### 3.3 Generate Release Notes

Create `release_notes_v1.X.Y.md`:

```markdown
# metal-candle v1.X.Y

[Compelling one-sentence summary]

## Highlights

- **Feature 1**: Description with user benefit
- **Feature 2**: Description with user benefit
- **Performance**: Specific claims from validated benchmarks

## Breaking Changes

(If any)

## New Features

(List with examples)

## Performance Improvements

(Validated claims from benchmark results)

## Bug Fixes

(List any fixes)

## Installation

\`\`\`toml
[dependencies]
metal-candle = "1.X"
\`\`\`

## Migration Guide

(If breaking changes)

## Contributors

- [@username](link) - description

## Full Changelog

See [CHANGELOG.md](CHANGELOG.md#1XY)
```

### 3.4 Review Documentation

```bash
# Build and review docs
cargo doc --all-features --workspace --no-deps --open

# Check for broken links
# Check all examples compile
# Verify README is up to date
```

**Documentation Checklist:**
- [ ] All public APIs documented
- [ ] Examples work and are referenced
- [ ] README reflects current capabilities
- [ ] CHANGELOG complete and accurate
- [ ] Release notes drafted

---

## Phase 4: Release Execution

### 4.1 Final Checks

```bash
# Ensure on main branch
git checkout main
git pull origin main

# All changes committed
git status  # Should be clean

# Final test suite
cargo test --all-features --workspace

# Final clippy
cargo clippy --all-features --workspace -- -D warnings
```

### 4.2 Create Release Commit

```bash
# Commit version updates
git add Cargo.toml CHANGELOG.md README.md
git commit -m "chore: release v1.X.Y"

# Create annotated tag
git tag -a v1.X.Y -m "Release v1.X.Y: [One-line summary]"

# Push commit and tag
git push origin main
git push origin v1.X.Y
```

### 4.3 Publish to crates.io

```bash
# Dry run first
cargo publish --dry-run --all-features

# Verify package contents
cargo package --list

# Publish (requires crates.io authentication)
cargo publish --all-features

# Verify publication
# Visit: https://crates.io/crates/metal-candle
```

**⚠️ Important:** Publishing is irreversible. Double-check everything!

### 4.4 Create GitHub Release

1. Go to: https://github.com/YOUR_USERNAME/metal-candle/releases
2. Click "Draft a new release"
3. Select tag: `v1.X.Y`
4. Release title: `v1.X.Y - [Descriptive Title]`
5. Copy release notes from `release_notes_v1.X.Y.md`
6. Attach benchmark results: `benchmark_results/YYYYMMDD_HHMMSS.tar.gz`
7. Click "Publish release"

---

## Phase 5: Post-Release

### 5.1 Verification

```bash
# Test installation from crates.io
cd /tmp
cargo new test_metal_candle
cd test_metal_candle
echo 'metal-candle = "1.X"' >> Cargo.toml
cargo build

# Verify docs.rs build
# Visit: https://docs.rs/metal-candle/1.X.Y
# (May take 10-30 minutes to build)
```

### 5.2 Announcements

Consider announcing on:
- [ ] GitHub Discussions
- [ ] Reddit: r/rust
- [ ] Rust Users Forum
- [ ] Discord: Rust ML channels
- [ ] Twitter/X (if applicable)
- [ ] HuggingFace Discord (if ML-related)

**Sample Announcement:**
```markdown
🚀 metal-candle v1.X.Y released!

[Compelling description]

Key improvements:
- Performance: [validated claim]
- New feature: [description]
- Enhanced: [description]

Install: cargo add metal-candle@1.X

Docs: https://docs.rs/metal-candle/1.X.Y
GitHub: https://github.com/YOUR_USERNAME/metal-candle/releases/tag/v1.X.Y

Benchmarks validated on Apple Silicon M1 Max.
```

### 5.3 Prepare for Next Release

```bash
# Create new [Unreleased] section in CHANGELOG
vim CHANGELOG.md
# Add:
# ## [Unreleased]
# 
# (Nothing yet)

git add CHANGELOG.md
git commit -m "docs: prepare CHANGELOG for next release"
git push origin main
```

---

## Emergency Rollback Procedure

If critical issues discovered after release:

### Option 1: Yank from crates.io

```bash
# Yank the problematic version
cargo yank --vers 1.X.Y

# Publish fixed version
# Update version to 1.X.Y+1
vim Cargo.toml
cargo publish
```

### Option 2: Patch Release

```bash
# Create hotfix branch
git checkout -b hotfix/v1.X.Y+1 v1.X.Y

# Fix issue
# ...

# Release as v1.X.Y+1
# Follow Phase 4 (expedited)
```

---

## Release Checklist Template

Use this checklist for each release:

```markdown
## Release v1.X.Y Checklist

### Pre-Release
- [ ] All tests passing
- [ ] Clippy clean (zero warnings)
- [ ] Coverage ≥80%
- [ ] Examples all work
- [ ] Dependencies audited

### Benchmarks
- [ ] Environment prepared (idle, powered, cool)
- [ ] Official benchmarks run (5+ runs)
- [ ] Results analyzed (variance <10%)
- [ ] BENCHMARKS.md updated
- [ ] Results committed to git

### Documentation
- [ ] CHANGELOG.md updated with version and date
- [ ] Version number updated in Cargo.toml
- [ ] README.md updated
- [ ] Release notes drafted
- [ ] API docs complete

### Release
- [ ] On main branch, pulled latest
- [ ] Release commit created
- [ ] Tag created: v1.X.Y
- [ ] Pushed to GitHub
- [ ] Published to crates.io
- [ ] GitHub Release created
- [ ] Benchmark results attached

### Post-Release
- [ ] Installation verified from crates.io
- [ ] docs.rs build successful
- [ ] Announcements posted (if applicable)
- [ ] CHANGELOG prepared for next version

### Sign-Off
- [ ] Release manager: _______________
- [ ] Date: _______________
- [ ] No critical issues: _______________
```

---

## Tools and Scripts

### Available Scripts

- `scripts/run_official_benchmarks.sh` - Official benchmark runner
- `scripts/check_release_readiness.sh` - Pre-release validation (TODO)
- `scripts/update_version.sh` - Automated version updates (TODO)

### Recommended Tools

```bash
# Install helpful tools
cargo install cargo-deny      # Security audits
cargo install cargo-outdated  # Check outdated deps
cargo install cargo-llvm-cov  # Coverage reports
cargo install cargo-release   # Automated releases (optional)
```

---

## FAQ

**Q: How often should we release?**  
A: Follow semantic versioning:
- Patch (1.X.Y): Bug fixes, every 2-4 weeks
- Minor (1.X.0): New features, every 2-3 months
- Major (2.0.0): Breaking changes, as needed

**Q: What if benchmarks show regression?**  
A: Investigate before releasing:
1. Re-run benchmarks (might be environmental)
2. Profile to identify cause
3. Fix or document as known issue
4. Do not release with unexplained regressions

**Q: Can we skip official benchmarks for patch releases?**  
A: For bug-fix-only patches: Yes, smoke tests suffice  
For any performance changes: No, always run official benchmarks

**Q: What if we don't have access to the benchmark hardware?**  
A: Options:
1. Delay release until hardware available
2. Run on available hardware, document environment
3. Use GitHub Actions results with clear disclaimers

**Q: How do we handle security vulnerabilities?**  
A: Patch releases (1.X.Y+1) immediately:
1. Fix vulnerability
2. Run tests (skip full benchmark if unrelated)
3. Expedited release process
4. Security advisory on GitHub

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| 1.0 | 2024-12-10 | Initial release process |
| 1.1 | 2024-12-10 | Added benchmark requirements |

---

**Maintained by:** [@GarthDB](https://github.com/GarthDB)  
**Last Updated:** December 10, 2025  
**Related Docs:** [BENCHMARK_CI.md](BENCHMARK_CI.md), [BENCHMARKS.md](../BENCHMARKS.md), [CONTRIBUTING.md](../CONTRIBUTING.md)





