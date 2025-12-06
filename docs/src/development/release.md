# Release Process

How new versions are released.

## Versioning

metal-candle follows [Semantic Versioning](https://semver.org/):
- **Major** (x.0.0): Breaking API changes
- **Minor** (1.x.0): New features, backwards compatible  
- **Patch** (1.0.x): Bug fixes, backwards compatible

## Release Checklist

Before releasing:

- [ ] All tests passing (`cargo test`)
- [ ] Zero clippy warnings (`cargo clippy -- -D warnings`)
- [ ] Code coverage ≥80%
- [ ] CHANGELOG.md updated
- [ ] Version bumped in Cargo.toml
- [ ] Documentation complete
- [ ] Examples tested
- [ ] Benchmarks run (if performance changes)

## Publishing

```bash
# Verify package
cargo package --list
cargo publish --dry-run

# Create git tag
git tag -a v1.x.0 -m "Release v1.x.0"
git push origin v1.x.0

# Publish to crates.io
cargo publish

# Create GitHub Release
# Use tag v1.x.0, copy from CHANGELOG.md
```

## Post-Release

- Monitor crates.io
- Update documentation site
- Announce on relevant forums
- Respond to issues

## See Also

- [CONTRIBUTING.md](https://github.com/GarthDB/metal-candle/blob/main/CONTRIBUTING.md)
- [CHANGELOG.md](https://github.com/GarthDB/metal-candle/blob/main/CHANGELOG.md)
