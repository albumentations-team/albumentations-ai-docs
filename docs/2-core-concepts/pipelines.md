# Pipelines: Composing Multiple Augmentations

While individual [Transforms](./transforms.md) are useful, data augmentation often involves applying a *sequence* of different operations. Albumentations makes this easy using **Pipelines**.

A pipeline, defined using [`albumentations.Compose`](https://albumentations.ai/docs/api-reference/core/composition/#Compose), chains multiple transforms together. When the pipeline is called, it applies the contained transforms sequentially to the input data.

## Defining a Simple Pipeline

You define a pipeline by passing a list of instantiated transforms to `A.Compose`.

```python
import albumentations as A
import cv2
import numpy as np

# Assume 'image' is loaded as a NumPy array (e.g., 100x100x3)
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) # Dummy image

# 1. Define the pipeline
pipeline = A.Compose([
    A.HorizontalFlip(p=0.5), # 50% chance to flip
    A.RandomBrightnessContrast(p=0.8), # 80% chance to adjust brightness/contrast
    A.GaussianBlur(p=0.3), # 30% chance to blur
])

# 2. Apply the pipeline
transformed_data = pipeline(image=image)
transformed_image = transformed_data['image']

print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
# Note: Shape usually remains the same unless a spatial transform like Resize is used.
```

## Compose Parameters

While the `transforms` list is the only mandatory argument, `A.Compose` accepts several other parameters to configure its behavior, especially when dealing with multiple data targets or needing fine-grained control:

```python
compose = A.Compose(
    transforms,
    bbox_params=None,
    keypoint_params=None,
    additional_targets=None,
    p=1.0,
    is_check_shapes=True,
    strict=False,
    mask_interpolation=None,
    seed=None,
    save_applied_params=False,
)
```

*   **`transforms` (list):** (Required) The list of augmentation instances to apply sequentially.
*   **`bbox_params` (`A.BboxParams` | dict | None):** Configuration for handling bounding boxes. If provided, the pipeline can accept a `bboxes` argument during the call. See the [Object Detection Guide](../3-basic-usage/bounding-boxes-augmentations.md) for details. Default: `None`.
*   **`keypoint_params` (`A.KeypointParams` | dict | None):** Configuration for handling keypoints. If provided, the pipeline can accept a `keypoints` argument during the call. See the [Keypoint Augmentation Guide](../3-basic-usage/keypoint-augmentations.md) for details. Default: `None`.
*   **`additional_targets` (dict[str, str] | None):** Defines how to handle additional input arrays beyond the standard `image`, `mask`, etc. Maps a custom input name (e.g., `'image2'`) to a standard target type (`'image'` or `'mask'`). See the [Using Additional Targets Guide](../4-advanced-guides/additional-targets.md) for details. Default: `None`.
*   **`p` (float):** The probability that the *entire* composition of transforms is applied. If a random check (`< p`) fails, the input data is returned unchanged. Note this differs from the individual `p` of transforms *inside* the list. Default: `1.0` (always apply the sequence).
*   **`is_check_shapes` (bool):** If `True`, Albumentations performs checks to ensure the spatial dimensions (H, W, potentially D) of all input arrays (`image`, `mask`, `masks`, `volume`, `mask3d`, etc.) are consistent before applying transforms. Disable (`False`) only if you are certain about data consistency and need maximum speed, as it can catch potential errors early. Default: `True`.
*   **`strict` (bool):** If `True`, enables strict validation mode during the transform call:
    1.  Checks that all keyword arguments passed (e.g., `image=...`, `mask=...`, `my_target=...`) are recognized (either standard targets or defined in `additional_targets`).
    2.  Validates that transforms are not called with unsupported arguments (though this is less common with the standard structure).
    3.  Raises a `ValueError` if any validation fails.
    If `False`, unknown arguments are ignored. Useful for debugging pipeline configuration. Default: `False`.
*   **`mask_interpolation` (int | None):** If set to an OpenCV interpolation flag (e.g., `cv2.INTER_LINEAR`), this value *overrides* the interpolation method used for masks in all applicable geometric transforms within the pipeline. If `None`, each transform uses its default mask interpolation (usually `cv2.INTER_NEAREST`). Default: `None`.
*   **`seed` (int | None):** Controls the reproducibility of random augmentations *within this specific `Compose` instance*.
    *   Albumentations uses its own internal random state generators (`self.py_random`, `self.random_generator`), completely independent from global seeds (`random.seed()`, `np.random.seed()`).
    *   Setting `seed` to an integer initializes these internal generators deterministically. Two `Compose` instances created with the same transforms and the same `seed` will produce the exact same sequence of random outcomes when called repeatedly on the same input.
    *   If `seed=None` (default), the internal state is initialized randomly, leading to different results each time a new `Compose` instance is created or run.
    *   See the [Creating Custom Transforms Guide](../4-advanced-guides/creating-custom-transforms.md#reproducibility-and-random-number-generation) for how to use the seeded generators in custom transforms. Default: `None`.
*   **`save_applied_params` (bool):** If `True`, the dictionary returned by the `Compose` call will include an extra key `'applied_transforms'`. The value will be a list of dictionaries, where each dictionary contains the name of an applied transform and the exact parameters it used for that specific call. Useful for debugging, replay, or analysis. Default: `False`.

## How Probabilities Work in Pipelines

When a pipeline created with `A.Compose` is called:

1.  It iterates through the transforms in the list *in the order they were provided*.
2.  For *each* transform in the list, its individual `p` value is checked.
3.  If the random check for a specific transform passes (based on its `p`), that transform is applied to the *current state* of the data (which might have been modified by previous transforms in the pipeline).
4.  If the check fails, that specific transform is skipped, and the pipeline moves to the next one.

This means that on any given call to the pipeline, a different subset of the defined transforms might actually be applied, controlled by their individual probabilities.

**Important Note:** `A.Compose` itself does *not* have a top-level `p` parameter to control whether the *entire* pipeline runs or not. It always executes and iterates through its contained transforms, letting their individual `p` values determine if they activate.

*(More content to follow on applying to targets, advanced pipelines, etc.)*

## Advanced Composition Utilities

Beyond the basic sequential application in `Compose`, Albumentations offers several utilities to build more complex pipelines with conditional or altered execution flow.

### [`OneOf`](https://albumentations.ai/docs/api-reference/core/composition/#OneOf): Apply Exactly One Transform

[`A.OneOf`](https://albumentations.ai/docs/api-reference/core/composition/#OneOf) takes a list of transforms and applies *exactly one* of them, chosen randomly based on their individual `p` values (which are normalized within the `OneOf` block). The `OneOf` block itself also has a `p` parameter determining if *any* transform within it gets applied.

```python
import albumentations as A

pipeline = A.Compose([
    # Apply either GaussianBlur or MotionBlur, but not both.
    # The choice between them depends on their p values (here, equal chance).
    # This entire OneOf block has a 90% chance of executing.
    A.OneOf([
        A.GaussianBlur(p=1.0), # p=1.0 within OneOf means it's considered
        A.MotionBlur(p=1.0),   # p=1.0 within OneOf means it's considered
    ], p=0.9), # 90% chance to apply *one* of the above
    A.HorizontalFlip(p=0.5)
])

# When pipeline(image=...) is called:
# - 90% chance: Either GaussianBlur OR MotionBlur is applied (50/50 chance each).
# - 10% chance: Neither GaussianBlur nor MotionBlur is applied.
# - Independently, 50% chance HorizontalFlip is applied.
```

### [`SomeOf`](https://albumentations.ai/docs/api-reference/core/composition/#SomeOf): Apply a Random Subset of Transforms

[`A.SomeOf`](https://albumentations.ai/docs/api-reference/core/composition/#SomeOf) takes a list of transforms and a number `n`. If the `SomeOf` block itself is activated (based on its main `p` value), it randomly selects `n` transforms from the list and attempts to apply them sequentially.

**Key Features:**

*   **Number of transforms (`n`):** This can be a fixed integer or a tuple `(min_n, max_n)`. If it's a tuple, the number of transforms *selected* will be a random integer chosen uniformly from the range `[min_n, max_n]` (inclusive) on each call.
*   **Selection Method (`replace`):** By default (`replace=False`), transforms are selected *without* replacement. If `replace=True`, transforms are selected *with* replacement.
*   **Main Probability (`p`):** Controls the probability that the `SomeOf` block executes at all. If this check fails, none of the transforms inside are considered.
*   **Individual Probabilities:** Crucially, after `SomeOf` selects the transforms, it attempts to apply each one. **Each selected transform is only applied if its own individual `p` value passes.** The `p` values inside the list are *not* ignored after selection.

```python
import albumentations as A

pipeline = A.Compose([
    # Attempt to apply between 1 and 3 of the following transforms (inclusive).
    # Selection is without replacement (default).
    # This SomeOf block has a 90% chance of being entered.
    A.SomeOf([
        A.GaussianBlur(p=0.5), # If selected, 50% chance to apply
        A.RandomBrightnessContrast(p=1.0), # If selected, 100% chance to apply
        A.HueSaturationValue(p=0.8), # If selected, 80% chance to apply
        A.Sharpen(p=0.3), # If selected, 30% chance to apply
    ], n=(1, 3), p=0.9), # Select 1 to 3 transforms, 90% of the time
    A.HorizontalFlip(p=0.5)
])

# When pipeline(image=...) is called:
# - 90% chance:
#     1. SomeOf selects N transforms (N between 1 and 3) randomly from the list.
#     2. It then iterates through these N selected transforms.
#     3. For each selected transform, it applies it *only if* that transform's individual `p` value passes its random check.
# - 10% chance: None of the transforms in SomeOf are considered.
# - Independently, 50% chance HorizontalFlip is applied.
```

### [`OneOrOther`](https://albumentations.ai/docs/api-reference/core/composition/#OneOrOther): Apply One of Two Transforms/Blocks

[`A.OneOrOther`](https://albumentations.ai/docs/api-reference/core/composition/#OneOrOther) applies either its `first` transform (or composed block) or its `second` transform (or composed block), chosen randomly based on their `p` values. It provides a clearer structure for simple A/B choices compared to `OneOf` with two elements.

```python
import albumentations as A

pipeline = A.Compose([
    # 50% chance apply first (Flip), 50% chance apply second (Rotate)
    A.OneOrOther(
        first=A.HorizontalFlip(p=1.0), # p=1.0 means always apply if chosen
        second=A.Rotate(limit=30, p=1.0), # p=1.0 means always apply if chosen
        p=0.5 # Probability of applying the `first` transform
    ),
    A.RandomBrightnessContrast(p=0.8)
])

# When pipeline(image=...) is called:
# - 50% chance HorizontalFlip is applied.
# - 50% chance Rotate is applied.
# - Independently, 80% chance RandomBrightnessContrast is applied.
```

### [`RandomOrder`](https://albumentations.ai/docs/api-reference/core/composition/#RandomOrder): Apply Transforms in Random Sequence

[`A.RandomOrder`](https://albumentations.ai/docs/api-reference/core/composition/#RandomOrder) takes a list of transforms and applies them sequentially, but shuffles the *order* of application randomly on each call. The `RandomOrder` block itself does *not* have a `p` parameter; it always executes and shuffles its children.

```python
import albumentations as A

pipeline = A.Compose([
    # Apply Flip and BrightnessContrast, but in a random order each time.
    A.RandomOrder([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.8),
    ]),
    A.GaussianBlur(p=0.3)
])

# When pipeline(image=...) is called:
# - Order 1: Flip (if p=0.5 passes) -> BrightnessContrast (if p=0.8 passes)
# - Order 2: BrightnessContrast (if p=0.8 passes) -> Flip (if p=0.5 passes)
# The order is chosen randomly.
# - Independently, 30% chance GaussianBlur is applied *after* the RandomOrder block.
```

### [`SelectiveChannelTransform`](https://albumentations.ai/docs/api-reference/core/composition/#SelectiveChannelTransform): Apply Transforms to Specific Channels

[`A.SelectiveChannelTransform`](https://albumentations.ai/docs/api-reference/core/composition/#SelectiveChannelTransform) applies its contained transforms only to specific channels of the input image. You specify the channels (by index) to affect.

A common use case is working with multi-channel data like RGBA images where you might want to apply certain augmentations only to the RGB channels (indices 0, 1, 2) and leave the Alpha channel (index 3) untouched.

```python
import albumentations as A
import numpy as np

# Assume a 4-channel image (RGBA)
image_rgba = np.random.randint(0, 256, (100, 100, 4), dtype=np.uint8)

# Apply GaussianBlur only to the first three channels (RGB)
pipeline = A.Compose([
    A.SelectiveChannelTransform(
        A.GaussianBlur(p=1.0), # Transform to apply
        channels=[0, 1, 2], # Apply only to RGB channels
        p=1.0 # Probability of applying this selective transform
    ),
    A.HorizontalFlip(p=0.5) # Applied to all channels
])

transformed_data = pipeline(image=image_rgba)
transformed_image = transformed_data['image']

# When pipeline(image=...) is called:
# - GaussianBlur is applied *only* to the R, G, B channels.
# - Independently, 50% chance the *entire* RGBA image is flipped.
```

### [`Sequential`](https://albumentations.ai/docs/api-reference/core/composition/#Sequential): Group Transforms with a Single Probability

[`A.Sequential`](https://albumentations.ai/docs/api-reference/core/composition/#Sequential) applies a list of transforms sequentially, just like `A.Compose`. The key difference is that `Sequential` itself has a top-level `p` parameter. If the random check for `Sequential`'s `p` fails, *none* of the transforms inside it are applied.

This contrasts with `A.Compose`, which *always* executes and iterates through its contained transforms, letting their individual `p` values determine if they activate. `Compose` also handles the setup for applying augmentations consistently across multiple targets (images, masks, bounding boxes), while `Sequential` is a simpler wrapper primarily for grouping transforms under a single probability.

Use `Sequential` when you want to treat a whole sequence of operations as a single augmentation block that either runs entirely or not at all, based on one probability value.

```python
import albumentations as A

pipeline = A.Compose([
    # This whole sequence runs only 40% of the time.
    A.Sequential([
        A.HorizontalFlip(p=1.0), # Always apply *if* Sequential runs
        A.RandomBrightnessContrast(p=1.0) # Always apply *if* Sequential runs
    ], p=0.4), # 40% chance to run the Flip -> BrightnessContrast sequence
    A.GaussNoise(p=0.5)
])

# When pipeline(image=...) is called:
# - 40% chance: HorizontalFlip is applied, then RandomBrightnessContrast is applied.
# - 60% chance: Neither Flip nor BrightnessContrast is applied.
# - Independently, 50% chance GaussNoise is applied.
```

## Nested Compositions

The real power of these composition utilities (`OneOf`, `SomeOf`, `Sequential`, etc.) comes from nesting them within each other or within a main `A.Compose` block. This allows you to build highly customized and complex augmentation pipelines.

Here's an example combining several concepts:

```python
import albumentations as A
import cv2
import numpy as np

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

complex_pipeline = A.Compose([
    # --- Block 1: Basic Transform ---
    A.HorizontalFlip(p=0.5), # Applied 50% of the time

    # --- Block 2: Choose ONE of two sequences ---
    A.OneOf([
        # Sequence A: Blur then Sharpen (runs as a unit if chosen)
        A.Sequential([
            A.GaussianBlur(p=1.0), # Always applies if Sequence A is chosen
            A.Sharpen(p=1.0)       # Always applies if Sequence A is chosen
        ], p=1.0), # p=1.0 ensures it's considered by OneOf

        # Sequence B: Emboss (runs if chosen)
        A.Emboss(p=1.0) # Always applies if Sequence B is chosen

    ], p=0.8), # 80% chance to run EITHER Sequence A OR Sequence B (50/50 split)

    # --- Block 3: Apply SOME of these color transforms ---
    A.SomeOf([
        A.RandomBrightnessContrast(p=0.7), # If selected, 70% chance to apply
        A.HueSaturationValue(p=0.6),       # If selected, 60% chance to apply
        A.ColorJitter(p=0.5)               # If selected, 50% chance to apply
    ], n=(1, 2), p=0.9), # 90% chance to select 1 or 2, then apply based on their p

    # --- Block 4: Noise, maybe ---
    A.GaussNoise(p=0.2) # Applied 20% of the time, independently
])

# Applying the complex pipeline:
transformed_data = complex_pipeline(image=image)
transformed_image = transformed_data['image']

# --- Execution Flow Example ---
# When complex_pipeline(image=...) is called:
# 1. HorizontalFlip runs based on its p=0.5.
# 2. The OneOf block runs based on its p=0.8.
#    - If it runs, it randomly chooses either Sequential or Emboss (50/50).
#    - If Sequential is chosen, GaussianBlur and Sharpen are both applied.
#    - If Emboss is chosen, Emboss is applied.
# 3. The SomeOf block runs based on its p=0.9.
#    - If it runs, it selects 1 or 2 transforms randomly from its list.
#    - For each selected transform, it runs based on *that transform's* p value.
# 4. GaussNoise runs based on its p=0.2.
```

This demonstrates how you can chain and nest different composition logic blocks to control the probability and flow of your augmentations precisely.

## Where to Go Next?

Now that you understand how to build pipelines, consider these next steps:

-   **[Transforms](./transforms.md):** Explore the individual augmentation operations you can include in your pipelines.
-   **[Setting Probabilities](./probabilities.md):** Get a deeper understanding of how the `p` parameter works for both individual transforms and composition blocks.
-   **[Working with Targets](./targets.md):** Learn how pipelines consistently apply augmentations to images, masks, bounding boxes, and keypoints.
-   **[Basic Usage Examples](../3-basic-usage):** See complete code examples of pipelines applied to common computer vision tasks.
-   **[Advanced Guides](../4-advanced-guides):** Discover techniques like serialization or creating custom transforms to integrate into your pipelines.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Experiment with combining transforms in the interactive tool.
