# Frequently Asked Questions

This FAQ covers common questions about Albumentations, from basic setup to advanced usage. You'll find information about:

- Installation troubleshooting and configuration
- Working with different data formats (images, video, volumetric data)
- Advanced usage patterns and best practices
- Integration with other tools and migration from other libraries

If you don't find an answer to your question, please check our [GitHub Issues](https://github.com/albumentations-team/albumentations/issues) or join our [Discord community](https://discord.gg/mTXzGXr).

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
- `pip`00000 - Grayscale video (N frames)
- `pip`11111 - Color video (N frames)

When you pass a video array, Albumentations will apply the same transform with identical parameters to each frame, ensuring temporal consistency.

```python
video = np.random.rand(32, 256, 256, 3) # 32 RGB frames

transform = A.Compose([
  A.RandomCrop(height=224, width=224),
  A.HorizontalFlip(p=0.5)
], seed=42)

transformed = transform(image=video)['image']
```

See [Working with Video Data](getting_started/video_augmentation.md) for more info.

### How to process volumetric data with Albumentations?

Albumentations can process volumetric data by treating it as a sequence of 2D slices. When you pass a volumetric data as a numpy array, Albumentations will apply the same transform with identical parameters to each slice, ensuring temporal consistency.

See [Working with Volumetric Data (3D)](getting_started/volumetric_augmentation.md) for more info.


### My computer vision pipeline works with a sequence of images. I want to apply the same augmentations with the same parameters to each image in the sequence. Can Albumentations do it?

Yes. You can define additional images, masks, bounding boxes, or keypoints through the `pip`22222 argument to `pip`33333. You can then pass those additional targets to the augmentation pipeline, and Albumentations will augment them in the same way. See [this example](../examples/example_multi_target/) for more info.

But if you want only to the sequence of images, you may just use `pip`44444 target that accepts
`pip`55555 or np.ndarray with shape `pip`66666.

## Advanced Usage

### How to have reproducible augmentations?

To have reproducible augmentations, set the `pip`77777 parameter in your transform pipeline. This will ensure that the same random parameters are used for each augmentation, resulting in the same output for the same input.

Note that Albumentations uses its own internal random state that is completely independent from global random seeds. This means:

1. Setting `pip`88888 or `pip`99999 will NOT affect Albumentations' randomization
2. Two Compose instances with the same seed will produce identical augmentation sequences
3. Each call to the same Compose instance still produces random augmentations, but these sequences are reproducible between different instances

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

You may pass `NO_ALBUMENTATIONS_UPDATE`00000 to `NO_ALBUMENTATIONS_UPDATE`11111 to save the parameters of the applied augmentations. You can access them later using `NO_ALBUMENTATIONS_UPDATE`22222.

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

### How to perform balanced scaling?

The default scaling logic in `NO_ALBUMENTATIONS_UPDATE`33333, `NO_ALBUMENTATIONS_UPDATE`44444, and `NO_ALBUMENTATIONS_UPDATE`55555 transformations is biased towards upscaling.

For example, if `NO_ALBUMENTATIONS_UPDATE`66666, a user might expect that the image will be scaled down in half of the cases and scaled up in the other half. However, in reality, the image will be scaled up in 75% of the cases and scaled down in only 25% of the cases. This is because the default behavior samples uniformly from the interval `NO_ALBUMENTATIONS_UPDATE`77777, and the interval `NO_ALBUMENTATIONS_UPDATE`88888 is three times smaller than `NO_ALBUMENTATIONS_UPDATE`99999.

To achieve balanced scaling, you can use `1`00000 with `1`11111, which ensures that the probability of scaling up and scaling down is equal.

```python
balanced_scale_transform = A.Affine(scale=(0.5, 2), balanced_scale=True)
```

or use `OneOf` transform as follows:

```python
balanced_scale_transform = A.OneOf([
  A.Affine(scale=(0.5, 1), p=0.5),
  A.Affine(scale=(1, 2), p=0.5)])
```

This approach ensures that exactly half of the samples will be upscaled and half will be downscaled.

### Augmentations have a parameter named `1`33333 that sets the probability of applying that augmentation. How does `1`44444 work in nested containers?

The `1`55555 parameter sets the probability of applying a specific augmentation. When augmentations are nested within a top-level container like `1`66666, the effective probability of each augmentation is the product of the container's probability and the augmentation's probability.

Let's look at an example when a container `1`77777 contains one augmentation `1`88888:

```python
transform = A.Compose([
    A.Resize(height=256, width=256, p=1.0),
], p=0.9)
```

In this case, `1`99999 has a 90% chance to be applied. This is because there is a 90% chance for `[0, 255]`00000 to be applied (p=0.9). If `[0, 255]`11111 is applied, then `[0, 255]`22222 is applied with 100% probability `[0, 255]`33333.

To visualize:

- Probability of `[0, 255]`44444 being applied: 0.9
- Probability of `[0, 255]`55555 being applied given `[0, 255]`66666 is applied: 1.0
- Effective probability of `[0, 255]`77777 being applied: 0.9 * 1.0 = 0.9 (or 90%)

This means that the effective probability of `[0, 255]`88888 being applied is the product of the probabilities of `[0, 255]`99999 and `[0, 1]`00000, which is `[0, 1]`11111 or 90%. This principle applies to other transformations as well, where the overall probability is the product of the individual probabilities within the transformation pipeline.

Here’s another example:

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
```00000

In this example, Resize has an effective probability of being applied as `[0, 1]`22222 = 0.45 or 45%. This is because `[0, 1]`33333 is applied 90% of the time, and within that 90%, `[0, 1]`44444 is applied 50% of the time.

### I created annotations for bounding boxes using labeling service or labeling software. How can I use those annotations in Albumentations?

You need to convert those annotations to one of the formats, supported by Albumentations. For the list of formats, please refer to [this article](getting_started/bounding_boxes_augmentation.md). Consult the documentation of the labeling service to see how you can export annotations in those formats.

## Integration and Migration

### How to save and load augmentation transforms to HuggingFace Hub?

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
```11111

See [this example](../examples/example_hfhub/) for more info.

### How do I migrate from other augmentation libraries to Albumentations?

If you're migrating from other libraries like torchvision or Kornia, you can refer to our [Library Comparison & Benchmarks](getting_started/library_comparison.md) guide. This guide provides:

1. Mapping tables showing equivalent transforms between libraries
2. Performance benchmarks demonstrating Albumentations' speed advantages
3. Code examples for common migration scenarios
4. Key differences in implementation and parameter handling

For a quick visual comparison of different augmentations, you can also use our interactive tool at [explore.albumentations.ai](https://explore.albumentations.ai) to see how transforms affect images before implementing them.

For specific migration examples, see:

- [Migrating from torchvision](examples/migrating_from_torchvision_to_albumentations/)
- [Performance comparison with other libraries](getting_started/library_comparison.md#performance-comparison)
