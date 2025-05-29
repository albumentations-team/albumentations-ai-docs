# Transforms: The Building Blocks of Augmentation

In Albumentations, a **Transform** represents a single augmentation operation. Think of it as the basic building
block for modifying your data. Examples include operations like flipping an image horizontally, applying Gaussian
blur, or adjusting brightness and contrast.

Each transform encapsulates the logic for applying a specific change to the input data.

## Quick Reference

**Key Concepts:**
- **Single operation**: Each transform performs one specific augmentation
- **Probability control**: `p` parameter controls application likelihood (0.0 to 1.0)
- **Parameter sampling**: Random values chosen from specified ranges each time
- **Transform types**: Pixel transforms vs. Spatial transforms
- **Target compatibility**: Some transforms affect multiple data types (image, mask, bboxes, keypoints)

**Most Important Transforms to Start With:**
- **Resizing/Cropping**: [`A.RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop), [`A.RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop)
- **Basic geometric**: [`A.HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip)
- **Regularization**: [`A.CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout)
- **Scale/rotation**: [`A.Affine`](https://explore.albumentations.ai/transform/Affine)

**Common Usage Patterns:**
- **Always apply**: [`A.Resize`](https://explore.albumentations.ai/transform/Resize)`(height=224, width=224, p=1.0)`
- **Sometimes apply**: [`A.HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip)`(p=0.5)`
- **Parameter ranges**: [`A.RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast)`(brightness_limit=(-0.2, 0.3), p=0.8)`

## Basic Transform Usage

### Applying a Single Transform

Using a single transform is straightforward. You import it, instantiate it with specific parameters,
and then call it like a function, passing your data as keyword arguments.

```python
import albumentations as A
import cv2
import numpy as np

# Load or create an image (NumPy array)
# image = cv2.imread("path/to/your/image.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Dummy image

# 1. Instantiate the transform
transform = A.HorizontalFlip(p=1.0)  # p=1.0 means always apply

# 2. Apply the transform to the image
transformed_data = transform(image=image)
transformed_image = transformed_data['image']

print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
```

### Transform Return Format

**Important:** All transforms return a **dictionary**, not just the transformed image:

```python
# Wrong - this won't work
# transformed_image = transform(image)

# Correct - transforms return dictionaries
result = transform(image=image)
transformed_image = result['image']

# You can also unpack if you know what targets you're using
result = transform(image=image, mask=mask)
new_image = result['image']
new_mask = result['mask']
```

## Core Transform Concepts

### The Probability Parameter `p`

A crucial parameter for every transform is `p`. This controls the probability that the transform
will be applied when called.

**Probability Values:**
- **`p=1.0`**: The transform is always applied
- **`p=0.0`**: The transform is never applied
- **`p=0.5`**: The transform has a 50% chance of being applied each time it's called

```python
import albumentations as A

# Always flip
always_flip = A.HorizontalFlip(p=1.0)

# Sometimes flip (50% chance)
maybe_flip = A.HorizontalFlip(p=0.5)

# Rarely blur (10% chance)
rare_blur = A.GaussianBlur(p=0.1)
```

This randomness allows you to introduce variety into your augmentation pipeline.
See [Setting Probabilities](./probabilities.md) for more detailed coverage.

### Parameter Sampling and Ranges

Beyond the `p` probability, many transforms introduce variability by accepting **ranges** of values
for certain parameters, typically as a tuple `(min_value, max_value)`. When such a transform is applied
(based on its `p` value), it randomly samples a specific value from the provided range for that execution.

**Range Examples:**
```python
import albumentations as A

# Brightness adjustment: random value between -0.2 and +0.3
brightness_transform = A.RandomBrightnessContrast(
    brightness_limit=(-0.2, 0.3),  # Range
    contrast_limit=(-0.1, 0.1),    # Range
    p=1.0
)

# Rotation: random angle between -15 and +15 degrees
rotation_transform = A.Rotate(
    limit=(-15, 15),  # Range
    p=0.7
)

# Fixed value vs range
fixed_blur = A.GaussianBlur(blur_limit=3, p=1.0)        # Always sigma=3
random_blur = A.GaussianBlur(blur_limit=(1, 5), p=1.0)  # Random sigma between 1-5
```

**Key Point:** Each time the transform is called, new random values are sampled from the ranges,
creating different variations even with the same transform instance.

## Transform Categories

Understanding transform types helps you choose the right augmentations and predict their effects on your data.
Here's both the technical categorization and practical guidance on when to use different transforms.

### Technical Categories

#### Pixel Transforms

**What they do:** Modify only the pixel values of the image itself. They do **not** change the geometry
or spatial arrangement.

**Effect on targets:** These transforms **only affect image-like targets** (image, images, volumes).
They do **not** modify masks, bounding boxes, or keypoints.

**Common pixel transforms:**
- **Color adjustments**: [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast), [`HueSaturationValue`](https://explore.albumentations.ai/transform/HueSaturationValue), [`ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter)
- **Blur effects**: [`GaussianBlur`](https://explore.albumentations.ai/transform/GaussianBlur), [`MotionBlur`](https://explore.albumentations.ai/transform/MotionBlur), [`Defocus`](https://explore.albumentations.ai/transform/Defocus)
- **Noise**: [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise), [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise), [`MultiplicativeNoise`](https://explore.albumentations.ai/transform/MultiplicativeNoise)
- **Compression**: [`ImageCompression`](https://explore.albumentations.ai/transform/ImageCompression)

#### Spatial Transforms

**What they do:** Alter the spatial properties of the image – its geometry, size, or orientation.

**Effect on targets:** Because they change geometry, these transforms **affect all spatial targets**:
images, masks, bounding boxes, keypoints, and volumes. All targets are transformed consistently
to maintain alignment.

**Common spatial transforms:**
- **Flips**: [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip), [`VerticalFlip`](https://explore.albumentations.ai/transform/VerticalFlip)
- **Rotations**: [`Rotate`](https://explore.albumentations.ai/transform/Rotate), [`SafeRotate`](https://explore.albumentations.ai/transform/SafeRotate)
- **Resizing**: [`Resize`](https://explore.albumentations.ai/transform/Resize), [`RandomScale`](https://explore.albumentations.ai/transform/RandomScale), [`RandomSizedCrop`](https://explore.albumentations.ai/transform/RandomSizedCrop)
- **Geometric distortions**: [`Affine`](https://explore.albumentations.ai/transform/Affine), [`Perspective`](https://explore.albumentations.ai/transform/Perspective), [`ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform)
- **Cropping**: [`RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop), [`CenterCrop`](https://explore.albumentations.ai/transform/CenterCrop), [`BBoxSafeRandomCrop`](https://explore.albumentations.ai/transform/BBoxSafeRandomCrop)

### Practical Categories: Building Your Pipeline Step-by-Step

Here's a recommended order for adding transforms to your pipeline, based on effectiveness and safety:

#### 1. Essential Foundation (Start Here)

**Always start with these:**
- **Size normalization**: [`RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop), [`RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop), [`SmallestMaxSize`](https://explore.albumentations.ai/transform/SmallestMaxSize)
- **Basic invariances**: [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip) (safe for most natural images)

```python
# Essential starter pipeline
essential_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
])
```

#### 2. High-Impact Regularization (Add Next)

**Proven to improve generalization:**
- **Dropout variants**: [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout), [`RandomErasing`](https://explore.albumentations.ai/transform/RandomErasing)
- **Scale/rotation**: [`Affine`](https://explore.albumentations.ai/transform/Affine) (conservative ranges first)

```python
# Add regularization
improved_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),
])
```

#### 3. Domain-Specific Enhancements

**Choose based on your data:**

**For aerial/medical images with rotational symmetry:**
- [`SquareSymmetry`](https://explore.albumentations.ai/transform/SquareSymmetry): Applies all 8 rotations/flips of a square

**For color-sensitive tasks:**
- [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast): Simulate lighting variations
- [`ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter): Comprehensive color augmentation

**For robustness to blur/noise:**
- [`GaussianBlur`](https://explore.albumentations.ai/transform/GaussianBlur): Camera focus variations
- [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise): Sensor noise simulation

**For reducing color dependence:**
- [`ToGray`](https://explore.albumentations.ai/transform/ToGray): Force shape/texture learning
- [`ChannelDropout`](https://explore.albumentations.ai/transform/ChannelDropout): Partial color removal

#### 4. Advanced/Specialized (Use With Caution)

**For specific domains or when basic augmentations aren't enough:**
- **Medical imaging**: [`ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform), [`GridDistortion`](https://explore.albumentations.ai/transform/GridDistortion)
- **Weather simulation**: [`RandomSunFlare`](https://explore.albumentations.ai/transform/RandomSunFlare), [`RandomRain`](https://explore.albumentations.ai/transform/RandomRain), [`RandomFog`](https://explore.albumentations.ai/transform/RandomFog)
- **Domain adaptation**: [`FDA`](https://explore.albumentations.ai/transform/FDA), [`HistogramMatching`](https://explore.albumentations.ai/transform/HistogramMatching)

### Mixed Transforms

Some transforms combine both pixel and spatial modifications:

```python
# RandomSizedCrop: Spatial (cropping) + Pixel (resizing interpolation)
mixed_transform = A.RandomSizedCrop(
    min_max_height=(50, 100),
    height=80,
    width=80,
    p=1.0
)
```

*Explore [`RandomSizedCrop`](https://explore.albumentations.ai/transform/RandomSizedCrop) to see how it combines spatial and pixel effects.*

## Practical Examples

### Building Pipelines Incrementally

**Key principle:** Start simple and add complexity gradually, testing validation performance after each addition.

#### Step 1: Minimal Baseline

Start with the absolute essentials:

```python
import albumentations as A
import numpy as np

# Minimal pipeline - just size normalization and basic flip
baseline_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
])

# Test on your data
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
result = baseline_pipeline(image=image)
print(f"Original: {image.shape}, Augmented: {result['image'].shape}")
```

#### Step 2: Add Proven Regularization

Add transforms known to improve generalization:

```python
# Enhanced pipeline - add dropout and affine transforms
enhanced_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    # Regularization transforms
    A.CoarseDropout(
        max_holes=8, max_height=32, max_width=32,
        fill_value=0, p=0.5
    ),
    A.Affine(
        scale=(0.8, 1.2),      # Conservative scaling
        rotate=(-15, 15),      # Small rotations
        p=0.7
    ),
])
```

#### Step 3: Domain-Specific Additions

Add transforms based on your specific use case:

```python
# Domain-specific pipeline example (natural images)
domain_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.7),

    # Color robustness
    A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=0.6
    ),

    # Optional: noise/blur for robustness
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3), p=1.0),
        A.GaussNoise(var_limit=(10, 50), p=1.0),
    ], p=0.3),
])
```

### Working with Multiple Targets

When working with masks, bboxes, or keypoints, spatial transforms affect all targets consistently:

```python
import albumentations as A
import numpy as np

# Prepare data with multiple targets
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
mask = np.random.randint(0, 5, (300, 300), dtype=np.uint8)

# Spatial transform - affects both image and mask
spatial_pipeline = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    # Pixel transform - only affects image
    A.RandomBrightnessContrast(p=0.5),
])

result = spatial_pipeline(image=image, mask=mask)
print(f"Image shape: {result['image'].shape}")
print(f"Mask shape: {result['mask'].shape}")
print("Spatial alignment maintained between image and mask")
```

### Validation Strategy

Always validate your augmentation choices:

```python
import albumentations as A

# Test different augmentation strengths
conservative = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
])

moderate = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=4, max_height=24, max_width=24, p=0.4),
    A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

aggressive = A.Compose([
    A.RandomCrop(height=224, width=224, p=1.0),
    A.HorizontalFlip(p=0.7),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.6),
    A.Affine(scale=(0.7, 1.3), rotate=(-20, 20), p=0.7),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
    ], p=0.6),
])

# Test each pipeline and measure validation performance
# Keep the one that gives best results on your specific task
```

## Finding and Exploring Transforms

Albumentations offers a wide variety of transforms organized by category and functionality.

### Discovery Resources

**Visual Exploration:**
- **[Explore Albumentations](https://explore.albumentations.ai)**: Interactive tool with visual examples
- See real-time effects of different transforms and their parameters

**Documentation:**
- **[API Reference](https://albumentations.ai/docs/api-reference/)**: Complete parameter documentation
- **[Supported Targets by Transform](../reference/supported-targets-by-transform.md)**:
  Compatibility matrix showing which transforms work with which data types

### Transform Categories

**By Purpose:**
- **Spatial**: Geometric modifications (flip, rotate, crop, resize)
- **Pixel**: Color and texture modifications (brightness, blur, noise)
- **Weather**: Environmental effects (rain, snow, fog)
- **Perspective**: Camera and lens effects (perspective, fisheye)

**By Target Support:**
- **Image-only**: Only modify pixel values
- **Dual**: Support both images and masks
- **Multi-target**: Support images, masks, bboxes, and keypoints

## Best Practices

### The Incremental Approach

**Most Important Rule:** Don't add many transforms at once. Build your pipeline step-by-step:

1. **Start minimal**: Begin with just cropping/resizing and basic flips
2. **Add one category**: Test validation performance after each addition
3. **Monitor metrics**: If performance doesn't improve, remove or adjust
4. **Visualize results**: Always check that augmented images look realistic

```python
# ❌ Don't do this - too many transforms at once
overwhelming_pipeline = A.Compose([
    A.RandomCrop(224, 224), A.HorizontalFlip(), A.VerticalFlip(),
    A.Rotate(limit=45), A.Affine(scale=(0.5, 2.0)), A.Perspective(),
    A.CoarseDropout(), A.GaussianBlur(), A.GaussNoise(),
    A.RandomBrightnessContrast(), A.ColorJitter(), A.ToGray(),
    # ... many more
])

# ✅ Do this - build incrementally
step1 = A.Compose([A.RandomCrop(224, 224), A.HorizontalFlip(p=0.5)])
# Test step1, measure validation performance
step2 = A.Compose([A.RandomCrop(224, 224), A.HorizontalFlip(p=0.5), A.CoarseDropout(p=0.3)])
# Test step2, compare with step1
# Continue adding one transform type at a time
```

### Transform Selection Guidelines

1. **Start with proven basics**: [`RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop), [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip), [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout)
2. **Match your domain**: Aerial imagery benefits from [`SquareSymmetry`](https://explore.albumentations.ai/transform/SquareSymmetry), medical from [`ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform)
3. **Conservative parameters first**: Use small ranges initially (e.g., `rotate=(-10, 10)` before `rotate=(-45, 45)`)
4. **Consider your model**: Some architectures handle geometric augmentations better than others

### Parameter Tuning Strategy

1. **Start conservative**: Small rotation angles, moderate dropout sizes
2. **Use domain knowledge**: Don't rotate faces 180° for face recognition
3. **Test systematically**: Change one parameter at a time
4. **Monitor validation**: Stop if metrics plateau or degrade

**Example conservative → aggressive progression:**

```python
# Conservative (start here)
conservative = A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), p=0.3)

# Moderate (if conservative helps)
moderate = A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.5)

# Aggressive (only if moderate still helps)
aggressive = A.Affine(scale=(0.7, 1.4), rotate=(-30, 30), p=0.7)
```

### Performance Considerations

1. **Order matters**: Put expensive transforms (like [`ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform)) late in pipeline
2. **Crop early**: Process smaller images when possible - [`RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop) first saves computation
3. **Use OneOf**: Instead of many blur types, use `A.OneOf([GaussianBlur, MotionBlur, MedianBlur])`
4. **Consider caching**: For repeated experimentation with the same base augmentations

### Validation and Debugging

**Always visualize your pipeline output:**

```python
import matplotlib.pyplot as plt

def visualize_augmentations(pipeline, image, num_examples=4):
    """Show multiple augmentation results"""
    fig, axes = plt.subplots(1, num_examples + 1, figsize=(15, 3))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis('off')

    # Augmented examples
    for i in range(num_examples):
        augmented = pipeline(image=image)['image']
        axes[i+1].imshow(augmented)
        axes[i+1].set_title(f"Augmented {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

# Use it to check your pipeline
# visualize_augmentations(your_pipeline, sample_image)
```

**Key questions to ask:**
- Do augmented images still look realistic?
- Are important features preserved?
- Is the augmentation too aggressive for your task?
- Does validation performance improve with each addition?

### When to Use Advanced Transforms

**Use specialized transforms only when:**
- Basic augmentations don't provide enough variation
- You have domain-specific needs (medical distortions, weather effects)
- You've exhausted simpler options and need more regularization
- You have computational budget for expensive operations

**For comprehensive guidance on building effective pipelines, see [Choosing Augmentations for Model Generalization](../3-basic-usage/choosing-augmentations.md).** That guide provides detailed, step-by-step instructions for selecting and combining transforms for maximum effectiveness.

## Where to Go Next?

Now that you understand the fundamentals of transforms and how to approach building augmentation pipelines:

**Essential Next Step:**
-   **[Choosing Augmentations for Model Generalization](../3-basic-usage/choosing-augmentations.md)**: **Start here!**
    Comprehensive, step-by-step guide for building effective augmentation pipelines. Covers the complete process from basic crops to advanced domain-specific transforms.

**Core Concepts:**
-   **[Pipelines](./pipelines.md)**: Learn how to combine transforms using `Compose`, `OneOf`, `SomeOf`, and other composition utilities
-   **[Probabilities](./probabilities.md)**: Deep dive into controlling transform application with the `p` parameter and probability calculations
-   **[Targets](./targets.md)**: Understand how transforms interact with images, masks, bboxes, keypoints, and volumes

**Practical Application:**
-   **[Task-Specific Guides](../3-basic-usage/)**: See transforms in action for classification, segmentation, detection, etc.
-   **[Performance Optimization](../3-basic-usage/performance-tuning.md)**: Make your augmentation pipelines fast and efficient

**Advanced Topics:**
-   **[Creating Custom Transforms](../4-advanced-guides/creating-custom-transforms.md)**: Build your own augmentations when built-in transforms aren't enough
-   **[Serialization](../4-advanced-guides/serialization.md)**: Save and load transform configurations for reproducible experiments

**Interactive Learning:**
-   **[Explore Transforms Visually](https://explore.albumentations.ai)**: **Upload your own images** and experiment with transforms to see their effects in real-time on your specific data
-   **[Transform Compatibility Reference](../reference/supported-targets-by-transform.md)**: Quick lookup for which transforms work with which data types

**Recommended Learning Path:**
1. Read [Choosing Augmentations](../3-basic-usage/choosing-augmentations.md) for practical guidance
2. Explore [Pipelines](./pipelines.md) to understand composition techniques
3. Apply transforms to your specific task using the [Basic Usage guides](../3-basic-usage/)
4. Optimize performance with the [Performance Tuning](../3-basic-usage/performance-tuning.md) guide
