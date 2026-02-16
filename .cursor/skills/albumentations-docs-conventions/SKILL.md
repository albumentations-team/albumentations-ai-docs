---
name: albumentations-docs-conventions
description: Enforces Markdown and image conventions for Albumentations AI docs. Use when editing docs, adding new pages, or adding images to ensure consistency with pre-commit checks.
---

# Albumentations Docs Conventions

## Markdown Rules

- **Filenames:** Use hyphens, not underscores (e.g. `oriented-bounding-boxes.md`)
- **Directories:** Use hyphens in path components under `docs/`
- **Relative links** (to other `.md` or dirs): Use hyphens in the path. Underscores allowed in link text and external URLs.
- **Auto-fix:** `python scripts/check_markdown.py --fix` fixes underscores in relative text links.

## Image Rules

- **Format:** All images in `img/` must be `.webp`
- **Relative image links:** May contain underscores in paths (exception to link rule)
- **Conversion:** Convert PNG/JPG to webp before adding: `cwebp input.png -o output.webp` or PIL/OpenCV

## Validation

```bash
pre-commit run --all-files
```

Runs markdown check (with --fix) and image format check. Fix any failures before committing.
