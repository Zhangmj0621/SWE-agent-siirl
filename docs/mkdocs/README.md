# Documentation Site

## Local Preview

```bash
# 1. Install dependencies
pip install "mkdocs>=1.6,<2" mkdocs-material mkdocs-glightbox mkdocs-static-i18n

# 2. Start dev server (with hot reload)
cd docs/mkdocs
mkdocs serve

# 3. Open browser
# http://127.0.0.1:8000
```

## Build Static Site

```bash
cd docs/mkdocs
mkdocs build
# Output in docs/mkdocs/site/
```

## GitHub Pages Deployment

Push to `master` branch triggers automatic deployment via `.github/workflows/docs.yml`.

Visit: `https://<org>.github.io/siirl-agentic/`
