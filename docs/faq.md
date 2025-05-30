# Frequently Asked Questions

This FAQ covers common questions about Albumentations, from basic setup to advanced usage. You'll find information about:

- Installation troubleshooting and configuration
- Working with different data formats (images, video, volumetric data)
- Advanced usage patterns and best practices
- Integration with other tools and migration from other libraries

If you don't find an answer to your question, please check our [GitHub Issues](https://github.com/albumentations-team/albumentations/issues) or join our [Discord community](https://discord.gg/AmMnDBdzYs).

## Installation

### I am receiving an error message `Failed building wheel for imagecodecs` when I am trying to install Albumentations. How can I fix the problem?

Try to update `pip` by running the following command:

```bash
python -m pip install --upgrade pip
```

### How to disable automatic checks for new versions?

To disable automatic checks for new versions, set the environment variable `NO_ALBUMENTATIONS_UPDATE` to `1`.

### How to make Albumentations use one CPU core?

Albumentations do not use multithreading by default, but libraries it depends on (like opencv) may use multithreading. To make Albumentations use one CPU core, you can set the following environment variables:

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Disable OpenCV multithreading and OpenCL
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

### Experiencing slow performance with PyTorch DataLoader multi-processing?

Some users have reported performance issues when using Albumentations with PyTorch's DataLoader in a multi-processing setup. This can occur on certain hardware/software configurations because OpenCV (cv2), which Albumentations uses under the hood, may spawn multiple threads within each DataLoader worker process. These threads can potentially interfere with each other, leading to CPU blocking and slower data loading.

If you encounter this issue, you can try disabling OpenCV's internal multithreading and OpenCL acceleration by calling:

```python
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
```

This should be done at the beginning of your script or before creating the DataLoader. Note that this solution may not be necessary for all users, and you should only apply it if you're experiencing performance problems with your specific setup.


## Data Formats and Basic Usage
### Supported Image Types

Albumentations works with images of type uint8 and float32. uint8 images should be in the `[0, 255]` range, and float32 images should be in the `[0, 1]` range. If float32 images lie outside of the `[0, 1]` range, they will be automatically clipped to the `[0, 1]` range.

### Why do you call `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` in your examples?

[For historical reasons](https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/), OpenCV reads an image in BGR format (so color channels of the image have the following order: Blue, Green, Red). Albumentations uses the most common and popular RGB image format. So when using OpenCV, we need to convert the image format to RGB explicitly.

### How to have reproducible augmentations?

To have reproducible augmentations, set the `seed` parameter in your transform pipeline. This will ensure that the same random parameters are used for each augmentation, resulting in the same output for the same input.

```python
transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=42)
```

## Working with Different Data Types

### How to process video data with Albumentations?

Albumentations can process video data by treating it as a sequence of frames in numpy array format:
- `(N, H, W)` - Grayscale video (N frames)
- `(N, H, W, C)` - Color video (N frames)

When you pass a video array, Albumentations will apply the same transform with identical parameters to each frame, ensuring temporal consistency.

```python
video = np.random.rand(32, 256, 256, 3) # 32 RGB frames

transform = A.Compose([
  A.RandomCrop(height=224, width=224),
  A.HorizontalFlip(p=0.5)
], seed=137)

transformed = transform(image=video)['image']
```

See [Working with Video Data](./3-basic-usage/video-augmentation.md) for more info.

### How to process volumetric data with Albumentations?

Albumentations can process volumetric data by treating it as a sequence of 2D slices. When you pass a volumetric data as a numpy array, Albumentations will apply the same transform with identical parameters to each slice, ensuring temporal consistency.

See [Working with Volumetric Data (3D)](./3-basic-usage/volumetric-augmentation.md) for more info.

### Which transforms work with my data type?

Different transforms support different combinations of targets (images, masks, bboxes, keypoints, volumes). Before building your pipeline, check the [Supported Targets by Transform](./reference/supported-targets-by-transform.md) reference to ensure your chosen transforms are compatible with your data types.

This reference is particularly useful when:
- Working with multiple targets simultaneously (e.g., images + masks + bboxes)
- Planning volumetric (3D) augmentation pipelines
- Debugging "unsupported target" errors
- Choosing transforms for specific computer vision tasks

### My computer vision pipeline works with a sequence of images. I want to apply the same augmentations with the same parameters to each image in the sequence. Can Albumentations do it?

Yes. You can define additional images, masks, bounding boxes, or keypoints through the `additional_targets` argument to [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose). You can then pass those additional targets to the augmentation pipeline, and Albumentations will augment them in the same way. See [this example](https://www.albumentations.ai/docs/examples/example-multi-target/) for more info.

But if you want only to the sequence of images, you may just use `images` target that accepts
`list[numpy.ndarray]` or np.ndarray with shape `(N, H, W, C) / (N, H, W)`.

## Advanced Usage

### How to have reproducible augmentations?

To have reproducible augmentations, set the `seed` parameter in your transform pipeline. This will ensure that the same random parameters are used for each augmentation, resulting in the same output for the same input.

Note that Albumentations uses its own internal random state that is completely independent from global random seeds. This means:

1. Setting `np.random.seed()` or `random.seed()` will NOT affect Albumentations' randomization
2. Two [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) instances with the same seed will produce identical augmentation sequences
3. Each call to the same [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) instance still produces random augmentations, but these sequences are reproducible between different instances

Example of reproducible augmentations:
```python
# These two transforms will produce identical sequences
transform1 = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=137)

transform2 = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=137)

# This will NOT affect Albumentations randomization
np.random.seed(137)
random.seed(137)
```

### How can I find which augmentations were applied to the input data and which parameters they used?

You may pass `save_applied_params=True` to [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) to save the parameters of the applied augmentations. You can access them later using `applied_transforms`.

```python
transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.5),
    A.Normalize(),
], save_applied_params=True, seed=42)

transformed = transform(image=image)['image']

print(transform["applied_transforms"])
```

### How to apply CutOut augmentation?

Albumentations provides the [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout) transform, which is a generalization of the [CutOut](https://arxiv.org/abs/1708.04552) and [Random Erasing](https://arxiv.org/abs/1708.04896) augmentation techniques. If you are looking for CutOut or Random Erasing, you should use [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout).

[`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout) generalizes these techniques by allowing:
- **Variable number of holes:** You specify a range for the number of holes (`num_holes_range`) instead of a fixed number or just one.
- **Variable hole size:** Holes can be rectangular, with ranges for height (`hole_height_range`) and width (`hole_width_range`). Sizes can be specified in pixels (int) or as fractions of image dimensions (float).
- **Flexible fill values:** You can fill the holes with a constant value (int/float), per-channel values (tuple), random noise per pixel (`'random'`), a single random color per hole (`'random_uniform'`), or using OpenCV inpainting methods (`'inpaint_telea'`, `'inpaint_ns'`).
- **Optional mask filling:** You can specify a separate `fill_mask` value to fill corresponding areas in the mask, or leave the mask unchanged (`None`).

Example using random uniform fill:
```python
import albumentations as A
import numpy as np

image = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)

# Apply CoarseDropout with 3-6 holes, each 10-20 pixels in size,
# filled with a random uniform color.
transform = A.CoarseDropout(
    num_holes_range=(3, 6),
    hole_height_range=(10, 20),
    hole_width_range=(10, 20),
    fill="random_uniform",
    p=1.0
)

augmented_image = transform(image=image)['image']
```

This transform randomly removes rectangular regions from an image, similar to CutOut, but with more configuration options.

To specifically mimic the original **CutOut** behavior, you can configure `CoarseDropout` as follows:
- Set `num_holes_range=(1, 1)` to always create exactly one hole.
- Set `hole_height_range` and `hole_width_range` to the same fixed value (e.g., `(16, 16)` for a 16x16 pixel square or `(0.1, 0.1)` for a square 10% of the image size).
- Set `fill=0` to fill the hole with black.

Example mimicking CutOut with a fixed 16x16 hole:
```python
cutout_transform = A.CoarseDropout(
    num_holes_range=(1, 1),
    hole_height_range=(16, 16),
    hole_width_range=(16, 16),
    fill=0,
    p=1.0 # Apply always, or adjust probability as needed
)
```

### How to perform balanced scaling?

The default scaling logic in [`RandomScale`](https://explore.albumentations.ai/transform/RandomScale), [`ShiftScaleRotate`](https://explore.albumentations.ai/transform/ShiftScaleRotate), and [`Affine`](https://explore.albumentations.ai/transform/Affine) transformations is biased towards upscaling.

For example, if `scale_limit = (0.5, 2)`, a user might expect that the image will be scaled down in half of the cases and scaled up in the other half. However, in reality, the image will be scaled up in 75% of the cases and scaled down in only 25% of the cases. This is because the default behavior samples uniformly from the interval `[0.5, 2]`, and the interval `[0.5, 1]` is three times smaller than `[1, 2]`.

To achieve balanced scaling, you can use [`Affine`](https://explore.albumentations.ai/transform/Affine) with `balanced_scale=True`, which ensures that the probability of scaling up and scaling down is equal.

```python
balanced_scale_transform = A.Affine(scale=(0.5, 2), balanced_scale=True)
```

or use [`OneOf`](https://www.albumentations.ai/docs/api-reference/core/composition/#OneOf) transform as follows:

```python
balanced_scale_transform = A.OneOf([
  A.Affine(scale=(0.5, 1), p=0.5),
  A.Affine(scale=(1, 2), p=0.5)])
```

This approach ensures that exactly half of the samples will be upscaled and half will be downscaled.

### How do I know which transforms support my combination of targets?

When working with multiple targets (images + masks + bboxes + keypoints), you need to ensure all transforms in your pipeline support your specific combination. The [Supported Targets by Transform](./reference/supported-targets-by-transform.md) reference shows exactly which transforms work with which target types.

This is especially important when:
- Building pipelines with bounding boxes or keypoints
- Working with volumetric (3D) data
- Debugging "unsupported target" errors
- Migrating from other libraries

For example, if you're working with images and bounding boxes, you'll need to use spatial-level (dual) transforms rather than pixel-level transforms, as pixel-level transforms don't modify bounding box coordinates.

### Augmentations have a parameter named `p` that sets the probability of applying that augmentation. How does `p` work in nested containers?

The `p` parameter sets the probability of applying a specific augmentation. When augmentations are nested within a top-level container like [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose), the effective probability of each augmentation is the product of the container's probability and the augmentation's probability.

Let's look at an example when a container [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) contains one augmentation [`Resize`](https://explore.albumentations.ai/transform/Resize):

```python
transform = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
], p=0.9)
```

In this case, [`Resize`](https://explore.albumentations.ai/transform/Resize) has a 90% chance to be applied. This is because there is a 90% chance for [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) to be applied (p=0.9). If [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) is applied, then [`Resize`](https://explore.albumentations.ai/transform/Resize) is applied with 100% probability `(p=1.0)`.

To visualize:

- Probability of [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) being applied: 0.9
- Probability of [`Resize`](https://explore.albumentations.ai/transform/Resize) being applied given [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) is applied: 1.0
- Effective probability of [`Resize`](https://explore.albumentations.ai/transform/Resize) being applied: 0.9 * 1.0 = 0.9 (or 90%)

This means that the effective probability of [`Resize`](https://explore.albumentations.ai/transform/Resize) being applied is the product of the probabilities of [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) and [`Resize`](https://explore.albumentations.ai/transform/Resize), which is `0.9 * 1.0 = 0.9` or 90%. This principle applies to other transformations as well, where the overall probability is the product of the individual probabilities within the transformation pipeline.

Here's another example:

```python
transform = A.Compose([
    A.Resize(height=256, width=256, p=0.5),
], p=0.9)
```

In this example, [`Resize`](https://explore.albumentations.ai/transform/Resize) has an effective probability of being applied as `0.9 * 0.5` = 0.45 or 45%. This is because [`Compose`](https://www.albumentations.ai/docs/api-reference/core/composition/#Compose) is applied 90% of the time, and within that 90%, [`Resize`](https://explore.albumentations.ai/transform/Resize) is applied 50% of the time.

### I created annotations for bounding boxes using labeling service or labeling software. How can I use those annotations in Albumentations?

You need to convert those annotations to one of the formats, supported by Albumentations. For the list of formats, please refer to [this article](./3-basic-usage/bounding-boxes-augmentations.md).

**Note**: Not all transforms support bounding boxes. Check the [Supported Targets by Transform](./reference/supported-targets-by-transform.md) reference to see which transforms work with your specific combination of data types.

Consult the documentation of the labeling service to see how you can export annotations in those formats.

## Troubleshooting and Performance

### My augmentations aren't working as expected. How do I debug them?

**Step 1: Visualize your pipeline output**
Always check what your augmentations are actually doing:

```python
# Remove Normalize and ToTensorV2 for visualization
vis_transform = A.Compose([
    t for t in your_transform.transforms
    if not isinstance(t, (A.Normalize, A.ToTensorV2))
])

# Apply and display
result = vis_transform(image=image)
plt.imshow(result['image'])
```

**Step 2: Check probabilities**
Low probabilities might make transforms seem inactive:
```python
# Force application for testing
test_transform = A.HorizontalFlip(p=1.0)  # Always apply
```

**Step 3: Verify transform compatibility**
Check the [Supported Targets by Transform](./reference/supported-targets-by-transform.md) to ensure your transforms support your data types.

**Step 4: Print transform parameters**
```python
# Add this to see what parameters are being sampled
transform.transforms[0].get_params()  # For specific transform debugging
```

### Why are my augmentations slow? How can I optimize performance?

**Priority fixes (biggest impact):**

1. **Crop early**: Place cropping transforms first in your pipeline
```python
# ✅ Good - process smaller images
A.Compose([
    A.RandomCrop(224, 224),  # First!
    A.HorizontalFlip(),
    A.GaussianBlur(),
])

# ❌ Bad - process full-size images unnecessarily
A.Compose([
    A.GaussianBlur(),
    A.RandomCrop(224, 224),  # Too late
])
```

2. **Use efficient image loading**: OpenCV is usually fastest
```python
# ✅ Fastest
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ❌ Slower
image = Image.open(path)
image = np.array(image)
```

3. **Optimize DataLoader**: Use multiple workers and pin memory
```python
DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

For comprehensive optimization strategies, see the [Performance Tuning Guide](./3-basic-usage/performance-tuning.md).

### I'm getting memory errors with large images. What should I do?

**For very large images (>4K resolution):**

1. **Crop aggressively early**: Use small crop sizes in your pipeline
2. **Process in patches**: Crop multiple smaller regions instead of resizing the whole image
3. **Use lower precision**: Convert to `float16` if your model supports it
4. **Reduce batch size**: Use gradient accumulation instead of large batches

```python
# Memory-efficient pipeline for large images
memory_efficient = A.Compose([
    A.RandomCrop(512, 512, p=1.0),  # Much smaller than original
    A.HorizontalFlip(p=0.5),
    # ... other transforms on smaller image
])
```

### I'm getting shape mismatch errors. How do I fix them?

**Common causes and solutions:**

1. **Multi-target shape mismatch**: All targets must have same spatial dimensions
```python
# ✅ Correct - same height/width
image = np.random.rand(256, 256, 3)  # (H, W, C)
mask = np.random.rand(256, 256)      # (H, W)

# ❌ Wrong - different spatial dimensions
image = np.random.rand(256, 256, 3)  # (H, W, C)
mask = np.random.rand(128, 128)      # Different (H, W)
```

2. **Volume/3D shape issues**: Check depth dimension alignment
```python
# ✅ Correct for 3D
volume = np.random.rand(64, 256, 256)    # (D, H, W)
mask3d = np.random.rand(64, 256, 256)    # Same (D, H, W)
```

3. **Bbox format confusion**: Verify your bbox format matches BboxParams
```python
# Make sure format matches your actual data
bbox_params = A.BboxParams(format='pascal_voc')  # [x_min, y_min, x_max, y_max]
# vs
bbox_params = A.BboxParams(format='coco')        # [x_min, y_min, width, height]
```

### How do I handle images with different aspect ratios efficiently?

**Strategy 1: Letterboxing (preserves all content)**
```python
A.Compose([
    A.LongestMaxSize(max_size=512),
    A.PadIfNeeded(min_height=512, min_width=512, border_mode=cv2.BORDER_CONSTANT),
])
```

**Strategy 2: Smart cropping (may lose some content)**
```python
A.Compose([
    A.SmallestMaxSize(max_size=512),
    A.RandomCrop(512, 512),
])
```

**Strategy 3: Aspect-ratio aware resizing**
```python
A.RandomResizedCrop(512, 512, scale=(0.8, 1.0), ratio=(0.75, 1.33))
```

## Integration and Migration

### How to save and load augmentation transforms to HuggingFace Hub?

```python
import albumentations as A
import numpy as np

transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.RGBShift(),
    A.Normalize(),
])

transform.save_pretrained("qubvel-hf/albu", key="train")
# The 'key' parameter specifies the context or purpose of the saved transform,
# allowing for organized and context-specific retrieval.
# ^ this will save the transform to a directory "qubvel-hf/albu"
# with filename "albumentations_config_train.json"

transform.save_pretrained("qubvel-hf/albu", key="train", push_to_hub=True)
# ^ this will save the transform to a directory "qubvel-hf/albu"
# with filename "albumentations_config_train.json"
# + push the transform to the Hub to the repository "qubvel-hf/albu"

transform.push_to_hub("qubvel-hf/albu", key="train")
# Use `save_pretrained` to save the transform locally and optionally push to the Hub.
# Use `push_to_hub` to directly push the transform to the Hub without saving it locally.
# ^ this will push the transform to the Hub to the repository "qubvel-hf/albu"
# (without saving it locally)

loaded_transform = A.Compose.from_pretrained("qubvel-hf/albu", key="train")
# ^ this will load the transform from local folder if exist or from the Hub
# repository "qubvel-hf/albu"
```

See [this example](https://www.albumentations.ai/docs/examples/example-hfhub/) for more info.

### How do I migrate from other augmentation libraries to Albumentations?

If you're migrating from other libraries like torchvision or Kornia, you can refer to our [Library Comparison](./torchvision-kornia2albumentations.md) guide. This guide provides:

1. Mapping tables showing equivalent transforms between libraries
2. Performance benchmarks demonstrating Albumentations' speed advantages
3. Code examples for common migration scenarios
4. Key differences in implementation and parameter handling

For a quick visual comparison of different augmentations, you can also use our interactive tool at [explore.albumentations.ai](https://explore.albumentations.ai) to see how transforms affect images before implementing them.

For specific migration examples, see:

- [Migrating from torchvision](https://www.albumentations.ai/docs/examples/migrating-from-torchvision-to-albumentations/)
- [Performance comparison with other libraries](./benchmarks)
