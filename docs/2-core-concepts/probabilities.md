# Setting Probabilities for Transforms in Augmentation Pipelines

Each augmentation in Albumentations has a parameter named `p` that controls the probability of applying that augmentation to input data. Understanding how probabilities work is essential for creating effective and predictable augmentation pipelines.

## Quick Reference

**Key Concepts:**
- **`p=1.0`**: Transform is always considered for application
- **`p=0.0`**: Transform is never considered for application
- **`p=0.5`**: Transform has a 50% chance of being considered
- **Pipeline probability**: Overall chance the entire pipeline runs
- **Nested probabilities**: How probabilities combine in composition blocks like `OneOf`

## Basic Probability Mechanics

### Individual Transform Probability

Setting `p=1` means the transform will always be considered for application, while `p=0` means it will never be considered. A value between 0 and 1 represents the chance it will be considered.

Some transforms default to `p=1`, while others default to `p=0.5`. Since default values can vary, it is recommended to explicitly set the `p` value for each transform in your pipeline to ensure clarity and avoid unexpected behavior.

### Probability in Practice

```python
import albumentations as A
import numpy as np

# Create a simple example
transform = A.Compose([
    A.HorizontalFlip(p=0.5),        # 50% chance to flip
    A.RandomBrightnessContrast(p=0.8),  # 80% chance to adjust brightness/contrast
    A.GaussianBlur(p=0.3),         # 30% chance to blur
])

# Each transform runs independently based on its p value
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
transformed = transform(image=image)
```

## Complex Probability Example

Let's examine a more complex pipeline to understand how probabilities interact:

```python
import albumentations as A
import numpy as np

# Define probabilities for clarity
prob_pipeline = 0.95
prob_rotate = 0.85
prob_oneof_noise = 0.75

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=prob_rotate),     # 85% chance to be considered
    A.OneOf([
        A.GaussNoise(p=0.9),             # 90% weight within OneOf block
        A.ISONoise(p=0.7),               # 70% weight within OneOf block
    ], p=prob_oneof_noise)               # 75% chance for OneOf block to run
], p=prob_pipeline,                      # 95% chance for entire pipeline to run
  seed=137)                             # Seed for reproducibility

# Apply the transform
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
transformed = transform(image=image)
transformed_image = transformed['image']

print("Transformation applied:", not np.array_equal(image, transformed_image))
```

## How Different Probability Levels Work

### Pipeline Probability (`prob_pipeline`)

The `p` parameter in `A.Compose` determines if *any* augmentations within it are applied:

- **`p=0.0`**: Pipeline never runs, input is always returned unchanged
- **`p=1.0`**: Pipeline always runs, inner transforms get a chance based on their own probabilities
- **`0 < p < 1`**: Pipeline runs with that specific probability

In our example (`prob_pipeline = 0.95`), the pipeline runs 95% of the time.

### Individual Transform Probability (`prob_rotate`)

Once the pipeline runs, each transform has its own probability:

- `RandomRotate90` with `p=0.85` has an 85% chance of being applied
- This is independent of other transforms in the pipeline

### Composition Block Probability (`prob_oneof_noise`)

Composition utilities like `OneOf` have their own probability layer:

- `OneOf` block with `p=0.75` has a 75% chance to run
- If it runs, it executes *exactly one* of its contained transforms

## Probability Calculations in `OneOf`

### How Selection Works

When a `OneOf` block runs, it normalizes the probabilities of inner transforms to determine selection weights:

**Example transforms in `OneOf`:**
- `GaussNoise(p=0.9)`
- `ISONoise(p=0.7)`

**Normalization process:**
1. Sum of probabilities: `0.9 + 0.7 = 1.6`
2. Normalized probability for `GaussNoise`: `0.9 ÷ 1.6 = 0.5625` (56.25%)
3. Normalized probability for `ISONoise`: `0.7 ÷ 1.6 = 0.4375` (43.75%)

### Selection Results

If the `OneOf` block runs:
- `GaussNoise` is selected 56.25% of the time
- `ISONoise` is selected 43.75% of the time

## Overall Probability Calculations

The actual probability of each transform being applied to the original image is the product of all probability layers:

### Mathematical Breakdown

| Transform | Calculation | Result |
|-----------|-------------|--------|
| `RandomRotate90` | `0.95 × 0.85` | **80.75%** |
| `GaussNoise` | `0.95 × 0.75 × 0.5625` | **40.08%** |
| `ISONoise` | `0.95 × 0.75 × 0.4375` | **31.15%** |

### Formula Pattern

```
Final Probability = Pipeline_p × Block_p × Normalized_Transform_p
```

## Edge Cases: When `p=1` Doesn't Change the Image

Even when a transform is applied (`p=1` or probability check succeeds), the image might not visually change in certain cases:

### Identity Operations

**Transforms with identity operations in their selection:**
- [`RandomRotate90`](https://explore.albumentations.ai/transform/RandomRotate90): Can select 0° rotation (no change)
- [`D4`](https://explore.albumentations.ai/transform/D4): Can select identity transformation
- [`RandomGridShuffle`](https://explore.albumentations.ai/transform/RandomGridShuffle): Might shuffle back to original positions

### Identity Parameter Sampling

**Geometric transforms can sample identity parameters:**
- [`Affine(rotate=(-10, 10))`](https://explore.albumentations.ai/transform/Affine): Might sample rotation = 0°
- [`ShiftScaleRotate`](https://explore.albumentations.ai/transform/ShiftScaleRotate): Could sample shift=0, scale=1, rotate=0

**Example:**
```python
# This always applies identity transformation
A.Affine(rotate=0, scale=1, translate_px=0, p=1)

# This might randomly sample identity parameters
A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), p=1)
```

> **Key Point:** The `p` parameter controls whether a transform *runs*, but the transform's internal
> logic determines whether that execution *visually changes* the image.

## Best Practices

### Probability Setting Guidelines

1. **Be Explicit**: Always set `p` values explicitly rather than relying on defaults
2. **Consider Independence**: Remember that transform probabilities are independent within `Compose`
3. **Calculate Overall Effects**: Use the multiplication rule to understand final probabilities
4. **Test Your Pipeline**: Verify that your probability settings achieve the desired augmentation frequency

### Common Patterns

```python
# Light augmentation (conservative)
light_transform = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(p=0.2),
    A.GaussianBlur(p=0.1),
])

# Moderate augmentation (balanced)
moderate_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),
        A.HueSaturationValue(p=1.0),
    ], p=0.3),
    A.GaussianBlur(p=0.2),
])

# Heavy augmentation (aggressive)
heavy_transform = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.RandomBrightnessContrast(p=0.6),
    A.OneOf([
        A.GaussianBlur(p=1.0),
        A.MedianBlur(p=1.0),
        A.MotionBlur(p=1.0),
    ], p=0.4),
])
```

## Where to Go Next?

Understanding probabilities is crucial for controlling your augmentation pipelines. Now you can:

-   **[Review Pipelines](./pipelines.md):** See how probabilities function within different composition utilities like `Compose`, `OneOf`, `SomeOf`, and `Sequential`.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Experiment with different augmentations, their parameters, and consider the impact of their `p` values.
-   **[See Basic Usage Examples](../3-basic-usage/index.md):** Look at practical code applying pipelines with specific probabilities for different tasks.
-   **[Learn How to Pick Augmentations](../3-basic-usage/choosing-augmentations.md):** Get insights into choosing appropriate transforms and their probabilities.
-   **[Understand Reproducibility](../4-advanced-guides/creating-custom-transforms.md#reproducibility-and-random-number-generation):** Learn how seeds interact with probabilities to ensure consistent results when needed.
