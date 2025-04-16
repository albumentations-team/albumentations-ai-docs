# Transforms: The Building Blocks of Augmentation

In Albumentations, a **Transform** represents a single augmentation operation. Think of it as the basic building block for modifying your data. Examples include operations like flipping an image horizontally, applying Gaussian blur, or adjusting brightness and contrast.

Each transform encapsulates the logic for applying a specific change to the input data.

## Applying a Single Transform

Using a single transform is straightforward. You import it, instantiate it (potentially with specific parameters), and then call it like a function, passing your data as keyword arguments.

```python
import albumentations as A
import cv2
import numpy as np

# Load or create an image (NumPy array)
# image = cv2.imread("path/to/your/image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8) # Dummy image

# 1. Instantiate the transform
transform = A.HorizontalFlip(p=1.0) # p=1.0 means always apply

# 2. Apply the transform to the image
transformed_data = transform(image=image)
transformed_image = transformed_data['image']

# You can see the original and transformed image
# import matplotlib.pyplot as plt
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original")
# plt.subplot(1, 2, 2)
# plt.imshow(transformed_image)
# plt.title("Transformed (Flipped)")
# plt.show()

print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
```

## The Probability Parameter `p`

A crucial parameter for every transform is `p`. This controls the probability that the transform will be applied when called.

-   `p=1.0`: The transform is always applied.
-   `p=0.0`: The transform is never applied.
-   `p=0.5`: The transform has a 50% chance of being applied each time it's called.

This allows you to introduce randomness into whether a specific augmentation happens. We cover probabilities in more detail on the [Setting Probabilities](./probabilities.md) page.

### Parameter Sampling

Beyond the `p` probability, many transforms introduce variability by accepting a *range* of values for certain parameters, typically as a tuple `(min_value, max_value)`. When such a transform is applied (based on its `p` value), it doesn't use a fixed parameter value. Instead, it randomly samples a specific value from the provided range for *that specific execution*.

For example, a transform like [`RandomBrightnessContrast`](https://explore.albumentations.ai/augmentation/RandomBrightnessContrast) might be configured with `brightness_limit=(-0.2, 0.3)`. Each time it's applied to an image, the actual brightness adjustment factor will be a different random number sampled uniformly from -0.2 to 0.3.

```python
import albumentations as A

# Initialize with a range for brightness adjustment
transform = A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.3), p=1.0)

# When transform(image=...) is called, the brightness factor will be sampled
# from the interval [-0.2, 0.3] for that specific call.
```

This dynamic sampling is another key mechanism for creating diverse augmentations from a single transform configuration.

## Categories of Transforms: Pixel vs. Spatial

Transforms in Albumentations can be broadly categorized based on what they affect:

1.  **Pixel-level Transforms:** These modify only the pixel values of the image itself. They do *not* change the geometry or spatial arrangement. Examples include [`GaussianBlur`](https://explore.albumentations.ai/augmentation/GaussianBlur), [`RandomBrightnessContrast`](https://explore.albumentations.ai/augmentation/RandomBrightnessContrast), [`HueSaturationValue`](https://explore.albumentations.ai/augmentation/HueSaturationValue), [`GaussNoise`](https://explore.albumentations.ai/augmentation/GaussNoise). Importantly, these transforms **do not affect associated targets** like masks, bounding boxes, or keypoints.

2.  **Spatial-level Transforms:** These alter the spatial properties of the image – its geometry. Examples include [`HorizontalFlip`](https://explore.albumentations.ai/augmentation/HorizontalFlip), [`Rotate`](https://explore.albumentations.ai/augmentation/Rotate), [`Affine`](https://explore.albumentations.ai/augmentation/Affine), [`Perspective`](https://explore.albumentations.ai/augmentation/Perspective), [`Resize`](https://explore.albumentations.ai/augmentation/Resize). Because they change the geometry, these transforms **also modify associated targets** like masks, bounding boxes, and keypoints to keep them synchronized with the image content.

Understanding this distinction is vital when working with tasks beyond simple classification.

## Finding Available Transforms

Albumentations offers a wide variety of transforms.

-   For a **conceptual overview and visual examples**, check out the [Explore Albumentations](https://explore.albumentations.ai) tool.
-   For a **comprehensive list** showing which transforms support which targets (masks, bboxes, keypoints, volumes), see the **[Supported Targets by Transform](../reference/supported-targets-by-transform.md)** reference page.
-   For detailed **API documentation** including all parameters for each transform, consult the main [API Reference](https://albumentations.ai/docs/api-reference/).

Now that you understand individual transforms, let's see how to combine them into [Pipelines](./pipelines.md).

## Where to Go Next?

With a grasp of individual transforms, you can:

-   **Learn about [Pipelines](./pipelines.md):** Combine multiple transforms into sequences using `Compose`, `OneOf`, etc.
-   **Understand [Probabilities](./probabilities.md):** Dive deeper into how the `p` parameter controls transform application.
-   **Study [Targets](./targets.md):** See how transforms interact with images, masks, bounding boxes, and keypoints.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Browse and experiment with the wide range of available augmentations.
-   **See [Basic Usage Examples](../3-basic-usage/index.md):** Look at practical code applying transforms and pipelines.
-   **Explore [Advanced Guides](../4-advanced-guides/index.md):** Learn about topics like creating your own custom transforms.
