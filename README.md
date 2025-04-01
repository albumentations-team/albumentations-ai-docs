# Albumentations AI Documentation Source

[![pre-commit.ci status](https://results.pre-commit.ci/latest/github/albumentations-team/albumentations-ai-docs/main.svg)](https://results.pre-commit.ci/latest/github/albumentations-team/albumentations-ai-docs/main.svg)

This repository contains the source Markdown files used to generate the documentation website for Albumentations AI, available at:

[https://albumentations.ai/docs/](https://albumentations.ai/docs/)

## Contribution

We welcome contributions to the documentation! To ensure consistency and quality, please adhere to the following guidelines when editing or adding content.

### Conventions

Markdown files in this repository follow specific conventions enforced by pre-commit hooks:

1.  **File Naming:** All Markdown filenames (`.md`) must use hyphens (`-`) instead of underscores (`_`).
2.  **Directory Naming:** All directory names within the `docs/` path must use hyphens (`-`) instead of underscores (`_`).
3.  **Relative Links:** Relative links (to other `.md` files or directories within this repository) must use hyphens (`-`) instead of underscores (`_`) in their paths. Underscores *are allowed* in link text (`[link_text]`) and in links to external websites or anchors (`#section_header`).
4.  **Image Links:**
    *   Relative image links *are allowed* to contain underscores in their paths.
    *   All relative image links *must* point to files with the `.webp` extension.

### Pre-commit Hooks

This repository uses `pre-commit` to automatically check and enforce coding standards and the Markdown conventions listed above.

**Setup:**

1.  Install `pre-commit`: `pip install pre-commit` (or `pip install -r requirements-dev.txt` if you have cloned the repo and activated a virtual environment).
2.  Install the git hooks: `pre-commit install`

Now, the hooks will run automatically on `git commit`.

**Running Hooks Manually:**

You can run the checks on all files at any time:

```bash
pre-commit run --all-files
```

**Autofixing:**

The Markdown convention hook is configured to automatically fix underscores in relative link paths. If the hook modifies files, simply `git add` the changes and commit again.

## Website Generation

(Information about the tool or process used to convert these Markdown files into the website https://albumentations.ai/docs/ can be added here.)
