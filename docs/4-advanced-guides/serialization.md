# Serialization of Augmentation Pipelines

Albumentations allows you to save the definition of your augmentation pipeline to a file and load it back later. This is useful for:

*   **Reproducibility:** Ensuring you can recreate the exact augmentation pipeline used for experiments.
*   **Sharing:** Easily sharing complex pipelines with collaborators.
*   **Configuration Management:** Storing pipeline definitions separately from code (e.g., in config files).
*   **Deployment:** Potentially loading predefined pipelines in production environments (with caveats).

## Core Functions: `A.save()` and `A.load()`

The primary functions for serialization are:

*   [`A.save(transform, filepath, data_format='json')`](https://albumentations.ai/docs/api-reference/core/serialization/#save): Saves an Albumentations transform (like `A.Compose` or a single transform) to a file.
*   [`A.load(filepath, data_format='json')`](https://albumentations.ai/docs/api-reference/core/serialization/#load): Loads an Albumentations transform from a file.

## Supported Formats

Albumentations supports two human-readable serialization formats:

*   **`json` (Default):** Uses Python's built-in `json` library. Produces standard JSON files. This is the default format.
*   **`yaml`:** Uses the `PyYAML` library (requires installation: `pip install pyyaml`). YAML is often considered slightly more human-readable than JSON.

The choice between them is largely preference, although JSON is more universal and requires no extra dependencies beyond Albumentations itself.

## Basic Example

Let's define a pipeline, save it, and load it back.

```python
import albumentations as A
import numpy as np
import cv2 # For drawing

# 1. Define the pipeline
transform = A.Compose([
    A.Resize(height=256, width=256),
    A.Rotate(limit=45, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

# 2. Save the pipeline to a JSON file
filepath_json = "./my_pipeline.json"
A.save(transform, filepath_json)

# You can inspect my_pipeline.json - it contains the transform definitions

# 3. Load the pipeline back from the file
loaded_transform_json = A.load(filepath_json)

# 4. Verify the loaded transform works
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
augmented_image = loaded_transform_json(image=image)['image']

print("Original shape:", image.shape)
print("Augmented shape:", augmented_image.shape)
print("Loaded transform type:", type(loaded_transform_json))

# --- Example using YAML ---
# Requires: pip install pyyaml
try:
    import yaml # Check if PyYAML is installed
    filepath_yaml = "./my_pipeline.yaml"
    A.save(transform, filepath_yaml, data_format='yaml')
    loaded_transform_yaml = A.load(filepath_yaml, data_format='yaml')
    print("YAML serialization successful.")
    # Verify loaded_transform_yaml works similarly...
except ImportError:
    print("PyYAML not installed, skipping YAML example.")

```

The loaded transform object will be identical in structure and parameters to the original one.

## Saving/Loading Pipelines with Parameters

Pipelines containing `bbox_params`, `keypoint_params`, or `additional_targets` are serialized correctly using `A.save()` and `A.load()`.

```python
import albumentations as A

# Pipeline with bbox_params and additional_targets
transform_complex = A.Compose([
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
   additional_targets={'image2': 'image'})

# Save and load
A.save(transform_complex, "complex_pipeline.json")
loaded_complex = A.load("complex_pipeline.json")

# Verify params are loaded
print("Loaded bbox_params:", loaded_complex.bbox_params)
print("Loaded additional_targets:", loaded_complex.additional_targets)
```

## Summary

*   Use `A.save()` and `A.load()` for saving/loading pipelines.
*   Choose between `'json'` (default, standard) and `'yaml'` (requires `PyYAML`).
*   Pipeline parameters (`bbox_params`, `keypoint_params`, `additional_targets`) are correctly serialized.
