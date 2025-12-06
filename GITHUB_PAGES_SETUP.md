# GitHub Pages Setup for metal-candle

This project uses GitHub Pages to host the mdBook documentation.

## Current Status

✅ **Workflow Configured**: `.github/workflows/docs.yml` is set up
⚠️ **Needs Manual Enable**: GitHub Pages must be enabled in repository settings

## How to Enable GitHub Pages

1. Go to your repository on GitHub: https://github.com/GarthDB/metal-candle

2. Click on **Settings** (top right)

3. Scroll down to **Pages** in the left sidebar

4. Under **Source**, select:
   - Source: **GitHub Actions**
   
5. Click **Save**

6. The workflow will automatically deploy on the next push to `main`

## Accessing Documentation

Once enabled, documentation will be available at:

https://garthdb.github.io/metal-candle/

## Testing the Workflow

After enabling Pages:

```bash
# Make any change to docs and push
git add docs/
git commit -m "docs: trigger pages deployment"
git push origin main
```

Watch the **Actions** tab on GitHub to see the deployment progress.

## Troubleshooting

### Pages Not Deploying

1. Check that GitHub Pages is enabled (see step 4 above)
2. Verify the **Actions** tab shows the workflow running
3. Check workflow permissions in Settings → Actions → General
   - Ensure "Read and write permissions" is selected
   - Enable "Allow GitHub Actions to create and approve pull requests"

### 404 Error

If you get a 404 after deployment:
- Wait 1-2 minutes for DNS propagation
- Clear browser cache
- Check the deployment URL in the Actions log

## Updating Documentation

The mdBook source is in `docs/src/`. To update:

1. Edit files in `docs/src/`
2. Test locally: `cd docs && mdbook serve`
3. Commit and push to `main`
4. GitHub Actions will automatically rebuild and deploy

## Local Preview

```bash
# Install mdBook
cargo install mdbook

# Serve locally
cd docs
mdbook serve --open
```

Documentation will be available at http://localhost:3000

