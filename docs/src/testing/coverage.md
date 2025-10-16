# Code Coverage

## Current Metrics

- **Overall Coverage**: 92.9%
- **Backend Module**: 92.3%
- **Error Types**: 100%
- **Minimum Requirement**: 80%

## Coverage Goals

### By Module

| Module | Target | Current | Status |
|--------|--------|---------|--------|
| Public APIs | 100% | TBD | 🚧 Phase 2+ |
| Backend | 90%+ | 92.3% | ✅ |
| Error Types | 100% | 100% | ✅ |
| LoRA Training | 100% | TBD | 🚧 Phase 3 |
| Inference | 100% | TBD | 🚧 Phase 4 |

## Measuring Coverage Locally

### Install LLVM Coverage Tools

```bash
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov
```

### Generate Coverage Report

```bash
# HTML report
cargo llvm-cov --all-features --workspace --html
open target/llvm-cov/html/index.html

# JSON report
cargo llvm-cov --all-features --workspace --json --output-path coverage.json

# Summary only
cargo llvm-cov --all-features --workspace --summary-only
```

## CI Enforcement

Coverage is enforced in CI:

``yaml
- name: Check coverage threshold (≥80%)
  run: |
    COVERAGE=$(jq -r '.data[0].totals.lines.percent' coverage.json)
    if awk "BEGIN {exit !($COVERAGE < 80)}"; then
      echo "❌ Coverage ${COVERAGE}% is below threshold of 80%"
      exit 1
    fi
```

PRs that drop coverage below 80% will fail.

## Platform-Specific Coverage

See [Platform Coverage Limits](./platform-limits.md) for details on why 100% coverage is not achievable on single-platform CI.

## Coverage Tools

- **cargo-llvm-cov**: Local coverage generation
- **Codecov**: CI coverage reporting and tracking
- **GitHub Actions**: Automated coverage checks

## Best Practices

1. **Write tests first** - TDD approach ensures high coverage
2. **Test edge cases** - Focus on error paths and boundaries
3. **Document untestable code** - Explain why certain code can't be tested
4. **Review coverage reports** - Check before submitting PRs

## Related

- [Testing Strategy](./strategy.md)
- [Platform Limits](./platform-limits.md)

