# Documentation Setup Summary

## Overview

metal-candle uses a **three-tier documentation strategy** following Rust best practices:

### 1. 📖 docs.rs - API Reference (Primary)
**URL**: https://docs.rs/metal-candle (once published)  
**Source**: `///` doc comments in source code  
**Built**: Automatically when published to crates.io  
**Purpose**: Complete API reference for all public types and functions

**Standards:**
- Every public item must have documentation
- Include examples that compile
- Document errors and panics
- Link to related types

**Example:**
```rust
/// Creates a new Metal device with the specified index.
///
/// # Examples
///
/// ```no_run
/// use metal_candle::Device;
/// let device = Device::new_metal(0)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`DeviceError::MetalUnavailable`] if Metal is not available.
pub fn new_metal(index: usize) -> Result<Self> {
    // ...
}
```

### 2. 📚 GitHub Pages - User Guide (Extended)
**URL**: https://garthdb.github.io/metal-candle/  
**Source**: `docs/` directory (mdBook)  
**Built**: Automatically on push to `main` via GitHub Actions  
**Purpose**: Tutorials, architecture docs, testing strategy, guides

**Contents:**
- Getting Started guides
- Architecture explanations
- Testing and quality documentation
- **Platform Coverage Limits** (answers questions like "why not 100% coverage?")
- Contributing guidelines
- Troubleshooting

**Build locally:**
```bash
cd docs
mdbook serve --open
```

### 3. 📝 README.md - Quick Reference
**URL**: https://github.com/GarthDB/metal-candle  
**Source**: Repository root  
**Purpose**: Quick start, project overview, badges, links

## Documentation Workflow

### For API Changes (Source Code)

1. Add/update `///` doc comments in source code
2. Run `cargo doc --open` to preview
3. Ensure examples compile with `cargo test --doc`
4. Commit with source changes

### For User Guide Changes

1. Edit files in `docs/src/`
2. Run `mdbook serve` to preview
3. Commit changes (GitHub Actions will deploy)

### For README Changes

1. Edit `README.md`
2. Keep it concise (quick reference only)
3. Link to extended docs for details

## Platform Coverage Documentation

The **Platform Coverage Limits** page addresses:
- Why 92.9% coverage instead of 100%
- What code is uncovered and why
- Platform-specific testing limitations
- Future plans for multi-platform CI

**Location**: `docs/src/testing/platform-limits.md`  
**URL**: https://garthdb.github.io/metal-candle/testing/platform-limits.html

## Deployment

### API Docs (docs.rs)
- **Trigger**: Publishing to crates.io
- **Manual**: `cargo doc --open` (local only)
- **Automatic**: Yes, via docs.rs infrastructure

### User Guide (GitHub Pages)
- **Trigger**: Push to `main` branch (paths: `docs/**` or `src/**/*.rs`)
- **Workflow**: `.github/workflows/docs.yml`
- **Manual**: Not needed (automatic deployment)
- **Preview**: `mdbook serve` (local only)

### README
- **Visible**: Immediately on GitHub
- **No build step**: Plain markdown

## Benefits of This Setup

✅ **Best Practice**: Follows standard Rust documentation patterns  
✅ **Automatic**: Docs.rs and GitHub Pages deploy automatically  
✅ **Discoverable**: Users know where to look (docs.rs, GitHub Pages)  
✅ **Maintainable**: Documentation lives with code  
✅ **Professional**: Industry-standard approach  

## Quick Links

- **Build API Docs**: `cargo doc --open`
- **Build User Guide**: `cd docs && mdbook serve`
- **Deploy**: Automatic on push to `main`
- **Edit User Guide**: Files in `docs/src/`
- **Edit API Docs**: `///` comments in `src/`

## Future Enhancements

- [ ] Add more examples to user guide
- [ ] Create tutorial videos (optional)
- [ ] Add benchmarking results page
- [ ] Multi-language support (if needed)

---

**Setup Complete**: Documentation infrastructure is ready!  
**Next Steps**: Write content as features are implemented in each phase.

