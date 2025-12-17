#!/bin/bash
# Update documentation metrics before release
# Run this manually before publishing a new version

set -e

echo "🔍 Extracting Current Metrics..."
echo ""

# Count tests
TEST_COUNT=$(cargo test --all-features --workspace -- --list 2>/dev/null | grep -E '^[^ ].*: test$' | wc -l | xargs)
echo "  Tests: $TEST_COUNT"

# Get coverage
echo "  Running coverage analysis (this may take a minute)..."
COVERAGE=$(LLVM_COV=/Users/garthdb/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/llvm-cov \
           LLVM_PROFDATA=/Users/garthdb/.rustup/toolchains/stable-aarch64-apple-darwin/lib/rustlib/aarch64-apple-darwin/bin/llvm-profdata \
           cargo llvm-cov --all-features --workspace --summary-only 2>&1 | \
           grep 'TOTAL' | awk '{for(i=1;i<=NF;i++) if($i ~ /%$/) {gsub(/%/,"",$i); print $i; exit}}')
echo "  Coverage: ${COVERAGE}%"

# Get version
VERSION=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "\(.*\)"/\1/')
echo "  Version: $VERSION"

echo ""
echo "📝 Current Documentation Values:"
echo ""

# Check README for test count
README_TESTS=$(grep -o '[0-9]\+ tests' README.md | head -1 | awk '{print $1}')
echo "  README.md test count: $README_TESTS"

# Check README for coverage
README_COV=$(grep -o '[0-9]\+\.[0-9]\+%' README.md | head -1)
echo "  README.md coverage: $README_COV"

echo ""

# Compare and suggest updates
if [ "$README_TESTS" != "$TEST_COUNT" ]; then
    echo "⚠️  Test count mismatch!"
    echo "   README has: $README_TESTS"
    echo "   Actual: $TEST_COUNT"
    echo ""
    echo "   Update README.md line ~24:"
    echo "   - **🛡️ Production Ready**: $TEST_COUNT tests, zero warnings, 100% API documentation"
    echo ""
fi

if [ "$README_COV" != "${COVERAGE}%" ]; then
    echo "⚠️  Coverage mismatch!"
    echo "   README has: $README_COV"
    echo "   Actual: ${COVERAGE}%"
    echo ""
    echo "   Update README.md Project Status section (~line 160):"
    echo "   **Coverage**: ${COVERAGE}% (exceeds 80% requirement)"
    echo ""
fi

# Check CHANGELOG has current version
if ! grep -q "## \[$VERSION\]" CHANGELOG.md; then
    echo "⚠️  CHANGELOG.md missing entry for version $VERSION"
    echo "   Add a section: ## [$VERSION] - $(date +%Y-%m-%d)"
    echo ""
fi

echo "✅ Metric check complete!"
echo ""
echo "To apply updates, edit the files manually or use sed:"
echo "  sed -i '' 's/$README_TESTS tests/$TEST_COUNT tests/' README.md"
echo "  sed -i '' 's/$README_COV/${COVERAGE}%/' README.md"







