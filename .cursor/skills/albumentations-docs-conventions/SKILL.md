---
name: albumentations-docs-conventions
description: Enforces Markdown and image conventions for Albumentations AI docs. Use when editing docs, adding new pages, or adding images to ensure consistency with pre-commit checks.
---

# Albumentations Docs Conventions

## Library

- The library is **AlbumentationsX** (package `albumentationsx`), not the old `albumentations` package.
- Install: `pip install albumentationsx`
- Import stays the same: `import albumentations as A`
- Run scripts with `python3.12` (the version where albumentationsx is installed) if the default `python3` points elsewhere.

## Markdown Rules

- **Filenames:** Use hyphens, not underscores (e.g. `oriented-bounding-boxes.md`)
- **Directories:** Use hyphens in path components under `docs/`
- **Relative links** (to other `.md` or dirs): Use hyphens in the path. Underscores allowed in link text and external URLs.
- **Auto-fix:** `python scripts/check_markdown.py --fix` fixes underscores in relative text links.

## Image Rules

- **Location:** All images live in project root `img/`, **not** `docs/img/`. Use subdirs matching doc structure: `img/getting_started/`, `img/introduction/`, `img/advanced/`.
- **Format:** All images in `img/` must be `.webp`
- **Relative image links:** From `docs/<section>/`, use `../../img/<subdir>/name.webp`. May contain underscores in paths (exception to link rule).
- **Conversion:** Convert PNG/JPG to webp before adding: `cwebp input.png -o output.webp` or PIL/OpenCV

## Rendering

- Markdown docs are rendered to the website using **Next.js**, not MkDocs.
- Some internal links (e.g. `../reference/supported-targets-by-transform.md`) may not exist as raw `.md` files in the repo — they are generated or resolved during the website build. Do not treat them as broken links.

## Validation

```bash
pre-commit run --all-files
```

Runs markdown check (with --fix) and image format check. Fix any failures before committing.
