# Using Additional Targets in Albumentations

## Motivation

Standard Albumentations pipelines seamlessly synchronize augmentations between an `image` and its corresponding `mask`, `bboxes`, or `keypoints`. However, sometimes you need to apply the exact same geometric transformations (and potentially other transforms) with identical random parameters to *multiple* image-like or mask-like arrays simultaneously.

Common use cases include:

*   **Stereo Images:** Applying the same crop, rotation, or flip to both left and right stereo images. (Note: If you only need identical processing for multiple *images*, you could alternatively stack them into a single NumPy array and pass them via the plural `images` argument, treating it like a video clip. See the [Video Augmentation Guide](./video-augmentation.md) for details on that approach.)
*   **Multi-modal Data:** Consistently transforming an image and its corresponding depth map or thermal map (treating them as image-like or mask-like depending on desired interpolation and color transform application).
*   **Multiple Masks:** Augmenting an image along with several different segmentation masks (e.g., foreground mask, parts mask) using the same geometric warp.
*   **Image Restoration Tasks:** Applying geometric transforms identically to a clean target image and a degraded input version, while applying degradation transforms (like noise or blur) only to the input.

Albumentations provides the `additional_targets` argument in `A.Compose` and the `add_targets` method specifically for these situations where multiple named arrays need synchronized augmentation parameters.

**Note on Other Multi-Input Scenarios:**

Albumentations has other mechanisms for different multi-input scenarios, covered in other guides:

*   For sequences (like video frames or volumetric slices) where the *same parameters* should apply across the sequence dimension, use the plural arguments (`images`, `masks`, `volumes`, `masks3d`). See the [Video](./video-augmentation.md) and [Volumetric](./volumetric-augmentation.md) guides.
*   For associating multiple labels or attributes with individual bounding boxes or keypoints, use `bbox_params` or `keypoint_params` with `label_fields`. See the [Bounding Box](../3-basic-usage/bounding-boxes-augmentations.md) and [Keypoint](../3-basic-usage/keypoint-augmentations.md) guides.

This guide focuses *only* on the `additional_targets` feature.

## Core Mechanism: `additional_targets` in `A.Compose`

Albumentations recognizes several standard target types that have specific processing rules:

*   `image`: The primary input image(s) (e.g., `[H, W, C]`). Receives geometric, color, and intensity transforms. Uses standard interpolation for geometric transforms.
*   `mask`: Segmentation mask(s) (e.g., `[H, W]`). Receives geometric transforms using nearest-neighbor interpolation. Does *not* receive color/intensity transforms.
*   `masks`: Multiple segmentation masks passed together (e.g., `[N, H, W]`). Processed like `mask`.
*   `bboxes`: Bounding boxes. Processed according to `bbox_params`. Requires `bbox_params` to be set.
*   `keypoints`: Keypoints. Processed according to `keypoint_params`. Requires `keypoint_params` to be set.
*   `volume`: A 3D volume (e.g., `[D, H, W, C]`). Receives 3D geometric transforms, and applicable 2D transforms slice-wise. Color/intensity transforms applied if treated as `'image'`.
*   `volumes`: Multiple 3D volumes (e.g., `[N, D, H, W, C]`). Processed like `volume` across the first dimension.
*   `mask3d`: A 3D mask (e.g., `[D, H, W]`). Receives 3D geometric transforms using nearest-neighbor interpolation. Does *not* receive color/intensity transforms.
*   `masks3d`: Multiple 3D masks (e.g., `[N, D, H, W]`). Processed like `mask3d` across the first dimension.

The `additional_targets` argument in `A.Compose` allows you to define **new keyword argument names** for your transform calls and **map them to the processing logic of one of the standard *single-instance* target types** (`'image'`, `'mask'`, `'volume'`, `'mask3d'`, etc.). This tells Albumentations: "Treat the data passed under this new name exactly like you would treat data passed under the standard target name specified in the mapping."

The `additional_targets` argument takes a dictionary:

*   **Keys:** Define the *new keyword argument names* you'll use when calling the transform (e.g., `'right_image'`, `'depth'`, `'mask2'`). These names **must not** clash with the standard names listed above.
*   **Values:** Specify the *standard target type* whose processing logic should be applied. You can map to `'image'`, `'mask'`, `'bboxes'`, or `'keypoints'` (and potentially volumetric types like `'volume'`, `'mask3d'`):
    *   Mapping to `'image'`: Applies geometric transforms (standard interpolation) and color/intensity transforms.
    *   Mapping to `'mask'`: Applies geometric transforms (nearest-neighbor interpolation) but not color/intensity transforms.
    *   Mapping to `'bboxes'`: Applies geometric transforms according to the rules defined in `bbox_params`. Requires `bbox_params` to be set in `Compose`.
    *   Mapping to `'keypoints'`: Applies geometric transforms according to the rules defined in `keypoint_params`. Requires `keypoint_params` to be set in `Compose`.

**Example: Augmenting Stereo Images**

```python
import albumentations as A
import numpy as np

# Treat 'right_image' exactly like the primary 'image'
transform = A.Compose([
    A.RandomResizedCrop(height=256, width=512, scale=(0.8, 1.0), p=1.0),
    A.ColorJitter(p=0.5),
    A.HorizontalFlip(p=0.5),
], additional_targets={'right_image': 'image'})

# Dummy stereo data
left_image = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)
right_image = np.random.randint(0, 256, (300, 600, 3), dtype=np.uint8)

# Apply transform, passing both images
augmented = transform(image=left_image, right_image=right_image)

aug_left = augmented['image']
aug_right = augmented['right_image']

print(f"Left shape: {aug_left.shape}, Right shape: {aug_right.shape}")
# Both images received the exact same crop parameters, color jitter values, and flip decision.
```

**Example: Augmenting Image and Multiple Masks**

```python
import albumentations as A
import numpy as np

# Define pipeline with two additional mask targets
transform = A.Compose([
    A.ShiftScaleRotate(p=0.5),
    A.RandomCrop(width=80, height=80, p=1.0),
    A.GaussianBlur(p=0.5) # Will only affect 'image'
], additional_targets={'semantic_mask': 'mask', 'parts_mask': 'mask'})

# Create dummy data
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
semantic_mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)
parts_mask = np.random.randint(0, 3, (100, 100), dtype=np.uint8)

# Apply transform passing all targets
augmented = transform(image=image,
                      semantic_mask=semantic_mask,
                      parts_mask=parts_mask)

aug_image = augmented['image']
aug_semantic_mask = augmented['semantic_mask']
aug_parts_mask = augmented['parts_mask']

print(f"Augmented Semantic Mask shape: {aug_semantic_mask.shape}")
print(f"Augmented Parts Mask shape: {aug_parts_mask.shape}")
# Image, semantic_mask, and parts_mask received identical ShiftScaleRotate/RandomCrop.
# Only image was potentially blurred by GaussianBlur.
```

## Dynamic Addition: `add_targets` Method

Add targets to an existing `Compose` object dynamically.

```python
import albumentations as A
import numpy as np

transform_dynamic = A.Compose([A.VerticalFlip(p=1.0)])

# Add targets later
new_targets = {'image2': 'image', 'mask_extra': 'mask'}
transform_dynamic.add_targets(new_targets)

# Prepare data
image1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
image2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
mask_extra = np.random.randint(0, 2, (100, 100), dtype=np.uint8)

# Call with dynamically added arguments
augmented = transform_dynamic(image=image1, image2=image2, mask_extra=mask_extra)

aug_image1 = augmented['image']
aug_image2 = augmented['image2']
aug_mask_extra = augmented['mask_extra']

print("Image1, Image2, and Mask_extra all vertically flipped identically.")
```

## Important Considerations

*   **Shape Consistency:** All input arrays passed in a single call to the transform (e.g., `image`, `right_image`, `depth`, `mask2`) **must** have the same spatial dimensions (Height and Width for 2D, Depth, Height, and Width for 3D) before transformations are applied. Albumentations validates this by default.
*   **Primary Use:** Best suited for multiple image-like or mask-like arrays needing synchronized parameters.
*   **Params Requirement:** If mapping to `'bboxes'` or `'keypoints'`, ensure the corresponding `bbox_params` or `keypoint_params` are provided to `Compose`.
*   **Naming:** Keys in `additional_targets` must be unique and not clash with standard arguments (`image`, `mask`, `masks`, `bboxes`, `keypoints`, `volume`, `volumes`, `mask3d`, `masks3d`).
*   **Transform Support:** Check individual transform documentation (`Targets` section) to confirm support for the target types you are using.

## When to Use Which Mechanism?

*   **Single image + associated mask/bboxes/keypoints:** Use standard arguments (`image`, `mask`, `bboxes`, `keypoints`) with `bbox_params`/`keypoint_params` if needed.
*   **Sequence/Batch (video, volumetric):** Use plural arguments (`images`, `masks`, `volumes`, `masks3d`) to apply identical parameters across the first dimension.
*   **Multiple Attributes per BBox/Keypoint:** Use `label_fields` within `bbox_params`/`keypoint_params` to handle multiple labels or flags per box/keypoint.
*   **Multiple Distinct Arrays (Image/Mask/BBox/Keypoint):** Use `additional_targets` to synchronize parameters across separate inputs when you need them treated according to a standard target processing logic (e.g., stereo pairs, multi-modal data, multiple masks, potentially multiple sets of bboxes/keypoints needing identical geometric transforms). Remember to provide `bbox_params` or `keypoint_params` if mapping to those types.

## Next Steps

*   See the [Compose API documentation](https://albumentations.ai/docs/api-reference/core/composition/#Compose) for details.
*   Explore the [Transforms Overview](../reference/supported-targets-by-transform.md) to see which transforms support which targets.

## Where to Go Next?

After learning how to use additional targets:

-   **Integrate into Task Pipelines:** Apply this technique in relevant basic usage scenarios like [Video Augmentation](../3-basic-usage/video-augmentation.md) (if treating frames separately), multi-modal data, or image restoration setups.
-   **Review Core Concepts:** Revisit [Targets](../2-core-concepts/targets.md) and [Pipelines](../2-core-concepts/pipelines.md) to solidify your understanding of how `A.Compose` manages different inputs.
-   **Explore Other Advanced Guides:**
    -   [Serialization](./serialization.md): Save pipelines that use additional targets.
    -   [Custom Transforms](./creating-custom-transforms.md): Learn how to handle additional targets within your own custom transforms.
-   **Consult API Documentation:** Refer to the [Compose documentation](https://albumentations.ai/docs/api-reference/core/composition/#Compose) and the [Supported Targets by Transform](../reference/supported-targets-by-transform.md) list for detailed information.
