---
name: validate-and-fix
description: After completing code changes, runs pytest and pre-commit hooks, then iteratively fixes any failures until all pass. Use when the user asks to run tests, validate changes, fix lint/test errors, or when finishing a coding task that should be verified.
---

# Validate and Fix (Albumentations Docs)

## Workflow

1. **Run tests:** `pytest`
2. **Run pre-commit:** `pre-commit run --all-files`
3. **If either fails:** Fix issues, re-run. Do not stop until both pass.

## Common Fixes

- **Markdown convention violations:** `python scripts/check_markdown.py --fix` on changed `.md` files, then re-add and commit
- **Image format (PNG/JPG in img/):** Convert to `.webp` before adding. No auto-fix.
- **Other pre-commit:** ruff, mypy, codespell—fix manually or with suggested tools

## Done When

Tests pass and pre-commit exits 0.
