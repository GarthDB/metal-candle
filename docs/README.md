# metal-candle Documentation

This directory contains the source for metal-candle's extended documentation, built with [mdBook](https://rust-lang.github.io/mdBook/).

## Documentation Architecture

### 📖 docs.rs (API Documentation)
**Location**: Automatically built from `///` doc comments in source code  
**URL**: https://docs.rs/metal-candle  
**Purpose**: Complete API reference for all public types, functions, and modules

### 📚 GitHub Pages (User Guide)
**Location**: This directory (`docs/`)  
**URL**: https://garthdb.github.io/metal-candle/  
**Purpose**: Tutorials, architecture docs, testing strategy, and guides

### 📝 README.md (Quick Reference)
**Location**: Repository root  
**Purpose**: Quick start, badges, and links to full documentation

## Building Locally

### Prerequisites

```bash
cargo install mdbook
```

### Build and Serve

```bash
cd docs
mdbook serve --open
```

This will:
1. Build the documentation
2. Start a local web server
3. Open your browser to http://localhost:3000
4. Auto-reload on changes

### Build Only

```bash
cd docs
mdbook build
```

Output will be in `docs/book/`.

## Contributing to Documentation

### Adding a New Page

1. Create the markdown file in `docs/src/`
2. Add it to `docs/src/SUMMARY.md`
3. Build and preview: `mdbook serve`
4. Commit both the new file and updated `SUMMARY.md`

### Writing Style

- **Clear and concise**: Get to the point quickly
- **Code examples**: Show, don't just tell
- **Links**: Cross-reference related sections
- **Update dates**: Not needed - git tracks changes

### Documentation Standards

- Use sentence case for headings
- Include code examples that actually compile
- Link to API docs for detailed reference
- Explain the "why" not just the "what"

## Deployment

Documentation is automatically built and deployed by GitHub Actions:

- **Trigger**: Push to `main` branch
- **Workflow**: `.github/workflows/docs.yml`
- **Deployment**: GitHub Pages

No manual deployment needed!

## Directory Structure

```
docs/
├── book.toml           # mdBook configuration
├── src/
│   ├── SUMMARY.md      # Table of contents
│   ├── introduction.md # Landing page
│   ├── guide/          # User guides
│   ├── architecture/   # Design docs
│   ├── testing/        # Testing & coverage
│   ├── development/    # Contributing
│   └── reference/      # Reference material
└── book/               # Build output (gitignored)
```

## Related

- [Contributing Guide](./src/development/contributing.md)
- [mdBook User Guide](https://rust-lang.github.io/mdBook/)

