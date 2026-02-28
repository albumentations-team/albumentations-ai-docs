---
name: notebook-to-doc-images
description: Workflow for generating doc images from Jupyter notebooks. Use when creating or updating example images from notebooks in temp/ for the Albumentations docs.
---

# Notebook to Doc Images

## Workflow

1. **Run notebook** in `temp/` (e.g. `example_obb_affine_boats.ipynb`)
2. **Save figures** to `img/<section>/<topic>/` (project root, not `docs/img/`) matching doc structure:
   - `img/getting_started/augmenting_obb/` for OBB examples
   - `img/getting_started/augmenting_bboxes/` for bbox examples
   - `img/getting_started/augmenting_keypoints/` for keypoints
   - `img/advanced/` for advanced guides (TTA, custom transforms, etc.)
3. **Export as .webp:** `plt.savefig(path, format='webp')` or convert after
4. **Link in docs:** Use relative path from doc file, e.g. `../../img/getting_started/augmenting_obb/name.webp`

## Example

```python
# In notebook
plt.savefig("../img/getting_started/augmenting_obb/obb_affine_single.webp", format="webp", bbox_inches="tight")
```

```markdown
<!-- In docs/3-basic-usage/oriented-bounding-boxes.md -->
![Description](../../img/getting_started/augmenting_obb/obb_affine_single.webp "Description")
```

## Validation

Run `pre-commit run --all-files` to ensure image format and markdown links pass.
