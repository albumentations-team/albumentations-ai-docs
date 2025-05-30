# Choosing Augmentations for Model Generalization

Selecting the right set of augmentations is key to helping your model generalize better to unseen data. While applying augmentations, it's also crucial to ensure your pipeline runs efficiently.

**Before diving into *which* augmentations to choose, we strongly recommend reviewing the guide on [Optimizing Augmentation Pipelines for Speed](./performance-tuning.md) to avoid CPU bottlenecks during training.**

This guide focuses on strategies for selecting augmentations that are *useful* for improving model performance.

## Quick Reference: The 7-Step Approach

**Build your pipeline incrementally in this order:**

1. **[Start with Cropping](#step-1-start-with-cropping-if-applicable)** - Size normalization first (always)
2. **[Basic Geometric Invariances](#step-2-add-basic-geometric-invariances)** - [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip), [`SquareSymmetry`](https://explore.albumentations.ai/transform/SquareSymmetry) for aerial
3. **[Dropout/Occlusion](#step-3-add-dropout--occlusion-augmentations)** - [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout), [`RandomErasing`](https://explore.albumentations.ai/transform/RandomErasing) (high impact!)
4. **[Reduce Color Dependence](#step-4-reduce-reliance-on-color-features)** - [`ToGray`](https://explore.albumentations.ai/transform/ToGray), [`ChannelDropout`](https://explore.albumentations.ai/transform/ChannelDropout) (if needed)
5. **[Affine Transformations](#step-5-introduce-affine-transformations-scale-rotate-etc)** - [`Affine`](https://explore.albumentations.ai/transform/Affine) for scale/rotation
6. **[Domain-Specific](#step-6-domain-specific-and-advanced-augmentations)** - Specialized transforms for your use case
7. **[Normalization](#step-7-final-normalization---standard-vs-sample-specific)** - Standard or sample-specific (always last)

**Essential Starter Pipeline:**
```python
A.Compose([
    A.RandomCrop(height=224, width=224),      # Step 1: Size
    A.HorizontalFlip(p=0.5),                  # Step 2: Basic geometric
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),  # Step 3: Dropout
    A.Normalize(),                            # Step 7: Normalization
])
```

**Key Principle:** Add one step at a time, test validation performance, keep what helps.

## Key Principles for Choosing Augmentations

Before building your pipeline, keep these points in mind:

*   **Add Incrementally:** Don't add dozens of transforms at once. Start with a basic set (like cropping and flips) and add new augmentations one by one or in small groups. Monitor your validation metric (loss, accuracy, F1-score, etc.) after adding each new augmentation. If the metric improves, keep it; if it worsens or stays the same, reconsider or remove it.
*   **Parameter Tuning is Empirical:** There's no single formula to determine the *best* parameters (e.g., rotation angle, dropout probability, brightness limit) for an augmentation. Choosing good parameters relies heavily on domain knowledge, experience, and experimentation. What works well for one dataset might not work for another.
*   **Visualize Your Augmentations:** Always visualize the output of your augmentation pipeline on sample images from *your* dataset. This is crucial to ensure the augmented images are still realistic and don't distort the data in ways that would harm learning. Use the interactive tool at [explore.albumentations.ai](https://explore.albumentations.ai/) to **upload your own images** and quickly test different transforms and parameters to see real-time results on your specific data.

## A Practical Approach to Building Your Pipeline

Instead of adding transforms randomly, here's a structured approach to build an effective pipeline:

### Step 1: Start with Cropping (If Applicable)

Often, the images in your dataset (e.g., 1024x1024) are larger than the input size required by your model (e.g., 256x256). Resizing or cropping to the target size should almost always be the **first** step in your pipeline. As highlighted in the performance guide, processing smaller images significantly speeds up all subsequent transforms.

*   **Training Pipeline:** Use [`A.RandomCrop`](https://explore.albumentations.ai/transform/RandomCrop) or [`A.RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop). If your original images might be *smaller* than the target crop size, ensure you set `pad_if_needed=True` within the crop transform itself (instead of using a separate `A.PadIfNeeded`).
    *   **Note on Classification:** For many image classification tasks (e.g., ImageNet training), [`A.RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop) is often preferred. It performs cropping along with potentially aggressive resizing (changing aspect ratios) and scaling, effectively combining the cropping step with some geometric augmentation. Using `RandomResizedCrop` might mean you don't need a separate `A.Affine` transform for scaling later.
*   **Validation/Inference Pipeline:** Typically use [`A.CenterCrop`](https://explore.albumentations.ai/transform/CenterCrop). Again, use `pad_if_needed=True` if necessary.

### Step 1.5: Alternative Resizing Strategies (`SmallestMaxSize`, `LongestMaxSize`)

Instead of directly cropping to the final `TARGET_SIZE`, two common strategies involve resizing based on the shortest or longest side first, often followed by padding or cropping.

*   **[`A.SmallestMaxSize`](https://explore.albumentations.ai/transform/SmallestMaxSize) (Shortest Side Resizing):**
    *   Resizes the image keeping the aspect ratio, such that the *shortest* side equals `max_size`.
    *   **Common Use Case (ImageNet Style Preprocessing):** This is frequently used *before* cropping in classification tasks. For example, resize with `SmallestMaxSize(max_size=TARGET_SIZE)` and then apply `A.RandomCrop(TARGET_SIZE, TARGET_SIZE)` or `A.CenterCrop(TARGET_SIZE, TARGET_SIZE)`. This ensures the image is large enough in its smaller dimension for the crop to extract a `TARGET_SIZE` x `TARGET_SIZE` patch without needing internal padding, while still allowing the crop to sample different spatial locations.

*   **[`A.LongestMaxSize`](https://explore.albumentations.ai/transform/LongestMaxSize) (Longest Side Resizing):**
    *   Resizes the image keeping the aspect ratio, such that the *longest* side equals `max_size`.
    *   **Common Use Case (Letterboxing/Pillarboxing):** This is often used when you need to fit images of varying aspect ratios into a fixed square input (e.g., `TARGET_SIZE` x `TARGET_SIZE`) *without* losing any image content via cropping. Apply `LongestMaxSize(max_size=TARGET_SIZE)` first. The resulting image will have one dimension equal to `TARGET_SIZE` and the other less than or equal to `TARGET_SIZE`. Then, apply `A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)` to pad the shorter dimension with a constant value (e.g., black) to reach the square `TARGET_SIZE` x `TARGET_SIZE`. This process is known as letterboxing (padding top/bottom) or pillarboxing (padding left/right).

```python
import albumentations as A
import cv2 # For border_mode constant

TARGET_SIZE = 256

# Example: SmallestMaxSize + RandomCrop (ImageNet style)
train_pipeline_shortest_side = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    # ... other transforms like HorizontalFlip ...
])

val_pipeline_shortest_side = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    # ... other transforms ...
])

# Example: LongestMaxSize + PadIfNeeded (Letterboxing)
pipeline_letterbox = A.Compose([
    A.LongestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.PadIfNeeded(min_height=TARGET_SIZE, min_width=TARGET_SIZE,
                  border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
    # ... other transforms ...
])
```

### Step 2: Add Basic Geometric Invariances

Next, add augmentations that reflect fundamental invariances in your data without distorting it unrealistically.

*   **Horizontal Flip:** [`A.HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip) is almost universally applicable for natural images (street scenes, animals, general objects like in ImageNet, COCO, Open Images). It reflects the fact that object identity usually doesn't change when flipped horizontally. The main exception is when directionality is critical and fixed, such as recognizing specific text characters or directional signs where flipping changes the meaning.
*   **Vertical Flip & 90/180/270 Rotations (Square Symmetry):** If your data is invariant to basic rotations and flips, incorporating these can be highly beneficial.
    *   For data invariant to axis-aligned flips and rotations by 90, 180, and 270 degrees (common in aerial/satellite imagery, microscopy, some medical scans), [`A.SquareSymmetry`](https://explore.albumentations.ai/transform/SquareSymmetry) is an excellent choice. It randomly applies one of the 8 symmetries of the square: identity, horizontal flip, vertical flip, diagonal flip (rotation by 180 + vertical flip), rotation 90 degrees, rotation 180 degrees, rotation 270 degrees, anti-diagonal flip (rotation by 180 + horizontal flip). A key advantage is that these are *exact* transformations, avoiding interpolation artifacts that can occur with arbitrary rotations (like those from `A.Rotate`).
    *   This type of symmetry can sometimes be useful even in unexpected domains. For example, in a [Kaggle competition on Digital Forensics](https://ieeexplore.ieee.org/abstract/document/8622031), identifying the camera model used to take a photo, applying `SquareSymmetry` proved beneficial, likely because sensor-specific noise patterns can exhibit rotational/flip symmetries.
    *   If *only* vertical flipping makes sense for your data, use [`A.VerticalFlip`](https://explore.albumentations.ai/transform/VerticalFlip) instead.

```python
import albumentations as A

TARGET_SIZE = 256

# Example for typical natural images
train_pipeline_step2_natural = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    # ... other transforms ...
])

# Example for aerial/medical images with rotational symmetry
train_pipeline_step2_aerial = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.SquareSymmetry(p=0.5), # Applies one of 8 symmetries
    # ... other transforms ...
])
```

### Step 3: Add Dropout / Occlusion Augmentations

Dropout, in its various forms, is a powerful regularization technique for neural networks. The core idea is to randomly remove parts of the input signal, forcing the network to learn more robust and diverse features rather than relying on any single dominant characteristic.

#### Available Dropout Transforms

Albumentations offers several transforms that implement this idea for images:

*   **[`A.CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout):** Randomly zeros out rectangular regions in the image.
*   **[`A.RandomErasing`](https://explore.albumentations.ai/transform/RandomErasing):** Similar to CoarseDropout, selects a rectangular region and erases its pixels (can fill with noise or mean values too).
*   **[`A.GridDropout`](https://explore.albumentations.ai/transform/GridDropout):** Zeros out pixels on a regular grid pattern.
*   **[`A.ConstrainedCoarseDropout`](https://explore.albumentations.ai/transform/ConstrainedCoarseDropout):** This is a powerful variant where dropout is applied *only* within the regions specified by masks or bounding boxes of certain target classes. Instead of randomly dropping squares anywhere, it focuses the dropout *on the objects themselves*.

#### Why Dropout Augmentation is Powerful

**1. Learning Diverse Features**
Imagine training an elephant detector. If the network only sees full images, it might heavily rely on the most distinctive feature, like the trunk. By using `CoarseDropout` or `RandomErasing`, sometimes the trunk will be masked out. Now, the network *must* learn to identify the elephant from its ears, legs, body shape, color, etc. This encourages the model to build a more comprehensive understanding of the object.

**2. Robustness to Occlusion**
In real-world scenarios, objects are often partially occluded. Training with dropout simulates this, making the model better at recognizing objects even when only parts are visible.

**3. Mitigating Spurious Correlations**
Models can sometimes learn unintended biases from datasets. Dropout can help by randomly removing potentially confounding elements, encouraging the network to focus on more fundamental features of the target object itself.

**4. Train Hard, Test Easy (Ensemble Effect)**
By training on images with randomly masked regions, the network learns to make predictions even with incomplete information – a harder task. During inference, the network sees the *complete*, unmasked image. This is analogous to how dropout layers work in neural network architectures.

#### Use Case Example: ConstrainedCoarseDropout

Imagine detecting small objects like sports balls in game footage. These objects might already be partially occluded. Standard `CoarseDropout` might randomly hit the ball too infrequently or only cover a tiny, insignificant part. With `ConstrainedCoarseDropout`, you can ensure that the dropout patches are specifically applied *within* the bounding box (or mask) of the ball class, more reliably simulating partial occlusion of the target object itself.

#### Recommendation

Consider adding [`A.CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout) or [`A.RandomErasing`](https://explore.albumentations.ai/transform/RandomErasing) to your pipeline, especially for classification and detection tasks. Start with moderate probabilities and dropout sizes, and visualize the results to ensure you aren't removing too much information.

```python
import albumentations as A

TARGET_SIZE = 256

# Example adding OneOf dropout transforms
train_pipeline_step3 = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        # Use ranges for number/size of holes
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), fill_value=0, p=1.0),
        # Use ratio and unit size range for grid
        A.GridDropout(ratio=0.5, unit_size_range=(0.1, 0.2), p=1.0)
    ], p=0.5), # Apply one of the dropouts 50% of the time
    # ... other transforms ...
])
```

### Step 4: Reduce Reliance on Color Features

Sometimes, color can be a misleading feature, or you might want your model to generalize across variations where color isn't a reliable indicator. Two transforms specifically target this:

*   **[`A.ChannelDropout`](https://explore.albumentations.ai/transform/ChannelDropout):** Randomly drops one or more channels from the input image (e.g., makes an RGB image RG, RB, GB, or single channel).
*   **[`A.ToGray`](https://explore.albumentations.ai/transform/ToGray):** Converts the image to grayscale.

**Why is this useful?**

1.  **Color Invariance:** If the color of an object isn't fundamental to its identity, these transforms force the network to learn shape, texture, and context cues instead. Consider classifying fruit: you want the model to recognize an apple whether it's red or green. `ToGray` removes the color information entirely for some training samples, while `ChannelDropout` removes partial color information, forcing reliance on shape.

2.  **Generalizing Across Conditions:** Real-world lighting can drastically alter perceived colors. Training with `ToGray` or `ChannelDropout` can make the model more robust to these variations.

3.  **Focusing on Texture/Shape:** For tasks where fine-grained texture or shape is critical (e.g., medical image analysis, defect detection), reducing the influence of color channels can sometimes help the model focus on these structural patterns.

**Recommendation:** If color is not a consistently reliable feature for your task, or if you need robustness to color variations, consider adding [`A.ToGray`](https://explore.albumentations.ai/transform/ToGray) or [`A.ChannelDropout`](https://explore.albumentations.ai/transform/ChannelDropout) with a moderate probability.

```python
import albumentations as A

TARGET_SIZE = 256

# Example adding OneOf color dropout transforms
train_pipeline_step4 = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    # Assuming previous dropout step is also included:
    A.OneOf([
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), fill_value=0, p=1.0),
        A.GridDropout(ratio=0.5, unit_size_range=(0.1, 0.2), p=1.0)
    ], p=0.5),
    # Now, apply one of the color-reducing transforms:
    A.OneOf([
        A.ToGray(p=1.0), # p=1.0 inside OneOf
        A.ChannelDropout(p=1.0) # p=1.0 inside OneOf
    ], p=0.1), # Apply one of these 10% of the time
    # ... other transforms ...
])
```


### Step 5: Introduce Affine Transformations (Scale, Rotate, etc.)

After basic flips, the next common category involves geometric transformations like scaling, rotation, translation, and shear. [`A.Affine`](https://explore.albumentations.ai/transform/Affine) is a powerful transform that can perform all of these simultaneously.

*   **Common Use Cases:** While `Affine` supports all four, the most frequently beneficial components are **scaling** (zooming in/out) and **small rotations**.
    *   **Scaling:** Neural networks often struggle with scale variations. An object learned at one size might not be recognized well if significantly larger or smaller. Since objects appear at different scales in the real world due to distance, augmenting with scale changes helps the model become more robust. Consider a street view image: people close to the camera might occupy a large portion of the frame, while people further down the street could be orders of magnitude smaller. Training with scale augmentation helps the model handle such drastic variations.
        *   A common and relatively safe starting range for the `scale` parameter is `(0.8, 1.2)`.
        *   However, depending on the expected variations in your data, much wider ranges like `(0.5, 2.0)` are also frequently used.
        *   **Important Note on Sampling:** When using a wide, asymmetric range like `scale=(0.5, 2.0)`, sampling uniformly from this interval means that values greater than 1.0 (zoom in) will be sampled more often than values less than 1.0 (zoom out). This is because the length of the zoom-in sub-interval `[1.0, 2.0]` (length 1.0) is twice the length of the zoom-out sub-interval `[0.5, 1.0)` (length 0.5). Consequently, the probability of sampling a zoom-in factor (2/3) is double the probability of sampling a zoom-out factor (1/3).
        *   To ensure an equal probability of zooming in versus zooming out, regardless of the range, `A.Affine` provides the `balanced_scale=True` parameter. When set, it first randomly decides whether to zoom in or out (50/50 chance), and *then* samples uniformly from the corresponding sub-interval (e.g., either `[0.5, 1.0)` or `[1.0, 2.0]`).
        *   Using scale augmentation complements architectural approaches like Feature Pyramid Networks (FPN) or RetinaNet that also aim to handle multi-scale features.
    *   **Rotation:** Small rotations (e.g., `rotate=(-15, 15)`) can simulate slight camera tilts or object orientations.
    *   **Translation/Shear:** Less commonly used for general robustness, but can be relevant in specific domains.
*   **Why `Affine`?** Albumentations offers other related transforms like [`ShiftScaleRotate`](https://explore.albumentations.ai/transform/ShiftScaleRotate), [`SafeRotate`](https://explore.albumentations.ai/transform/SafeRotate), [`RandomScale`](https://explore.albumentations.ai/transform/RandomScale), and [`Rotate`](https://explore.albumentations.ai/transform/Rotate). However, `Affine` efficiently combines these operations, often making it a preferred choice for applying scale and rotation together.
    *   Additionally, [`A.Perspective`](https://explore.albumentations.ai/transform/Perspective) is another powerful geometric transform. While `Affine` performs parallel-preserving transformations, `Perspective` introduces non-parallel distortions, effectively simulating viewing objects or scenes from different angles or camera positions. It can be used in addition to or sometimes as an alternative to `Affine`, depending on the desired type of geometric variation.

```python
import albumentations as A

TARGET_SIZE = 256

# Example adding Affine after flips
train_pipeline_step5 = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Affine(
        scale=(0.8, 1.2),      # Zoom in/out by 80-120%
        rotate=(-15, 15),      # Rotate by -15 to +15 degrees
        # translate_percent=(0, 0.1), # Optional: translate by 0-10%
        # shear=(-10, 10),          # Optional: shear by -10 to +10 degrees
        p=0.7
    ),
    # Step 3: Dropout (example)
    A.OneOf([
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), fill_value=0, p=1.0),
        A.GridDropout(ratio=0.5, unit_size_range=(0.1, 0.2), p=1.0)
    ], p=0.5),
    # Step 4: Color Reduction (example)
    A.OneOf([
        A.ToGray(p=1.0),
        A.ChannelDropout(p=1.0)
    ], p=0.1),
    # ... other transforms ...
])
```

### Step 6: Domain-Specific and Advanced Augmentations

Once you have a solid baseline pipeline with cropping, basic invariances (like flips), dropout, and potentially color reduction and affine transformations, you can explore more specialized augmentations tailored to your specific domain, task, or desired model robustness. The augmentations below are generally applied *after* the initial cropping and geometric transforms.

#### Medical and Scientific Imaging

**Non-Linear Distortions:**
For domains where straight lines might deform (common in medical imaging like endoscopy or MRI due to tissue movement or lens effects):
- [`A.ElasticTransform`](https://explore.albumentations.ai/transform/ElasticTransform): Simulates tissue deformation
- [`A.GridDistortion`](https://explore.albumentations.ai/transform/GridDistortion): Non-uniform grid warping
- [`A.ThinPlateSpline`](https://explore.albumentations.ai/transform/ThinPlateSpline): Smooth deformation

**Histopathology (H&E Stain Augmentation):**
For histopathology images stained with Hematoxylin and Eosin (H&E), color variations due to staining processes are common:
- [`A.ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter): Simulate staining variations
- [`A.RandomToneCurve`](https://explore.albumentations.ai/transform/RandomToneCurve): Non-linear tone changes

#### Robustness Augmentations

**Color and Lighting Variations:**
To make your model less sensitive to lighting and color shifts:
- [`A.RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast): Basic lighting changes
- [`A.ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter): Comprehensive color augmentation
- [`A.RandomGamma`](https://explore.albumentations.ai/transform/RandomGamma): Gamma correction simulation
- [`A.HueSaturationValue`](https://explore.albumentations.ai/transform/HueSaturationValue): HSV color space adjustments
- [`A.PlanckianJitter`](https://explore.albumentations.ai/transform/PlanckianJitter): Color temperature variations (outdoor scenes)

**Noise Simulation:**
Simulate sensor noise or transmission errors:
- [`A.GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise): General Gaussian noise
- [`A.ISONoise`](https://explore.albumentations.ai/transform/ISONoise): Camera sensor noise at different ISO levels
- [`A.MultiplicativeNoise`](https://explore.albumentations.ai/transform/MultiplicativeNoise): Speckle-like noise
- [`A.SaltAndPepper`](https://explore.albumentations.ai/transform/SaltAndPepper): Random black/white pixels

**Blur Effects:**
Simulate motion blur, defocus, or atmospheric effects:
- [`A.GaussianBlur`](https://explore.albumentations.ai/transform/GaussianBlur): Out-of-focus simulation
- [`A.MotionBlur`](https://explore.albumentations.ai/transform/MotionBlur): Camera/object movement
- [`A.MedianBlur`](https://explore.albumentations.ai/transform/MedianBlur): Edge-preserving blur
- [`A.AdvancedBlur`](https://explore.albumentations.ai/transform/AdvancedBlur): Combines different blur types
- [`A.ZoomBlur`](https://explore.albumentations.ai/transform/ZoomBlur): Radial blur effect

**Compression and Quality Degradation:**
Simulate effects of saving images in lossy formats or resizing:
- [`A.Downscale`](https://explore.albumentations.ai/transform/Downscale): Downscale then upscale (loss of detail)
- [`A.ImageCompression`](https://explore.albumentations.ai/transform/ImageCompression): JPEG, WebP compression artifacts

#### Environmental and Weather Effects

**Outdoor Scene Simulation:**
- [`A.RandomSunFlare`](https://explore.albumentations.ai/transform/RandomSunFlare): Sun flare effects
- [`A.RandomShadow`](https://explore.albumentations.ai/transform/RandomShadow): Shadow simulation
- [`A.RandomFog`](https://explore.albumentations.ai/transform/RandomFog): Atmospheric fog
- [`A.RandomRain`](https://explore.albumentations.ai/transform/RandomRain): Rain effects
- [`A.RandomSnow`](https://explore.albumentations.ai/transform/RandomSnow): Snow simulation

> **Note:** While `RandomSunFlare` and `RandomShadow` were initially designed for street scenes, they have shown surprising utility in other domains like Optical Character Recognition (OCR), potentially by adding complex occlusions or lighting variations.

#### Specialized Applications

**Context Independence:**
- [`A.GridShuffle`](https://explore.albumentations.ai/transform/GridShuffle): If your task should not rely on spatial context between different parts of the image (e.g., certain texture analysis tasks)

**Spectrogram Augmentation:**
For spectrograms (visual representations of audio frequencies over time):
- [`A.XYMasking`](https://explore.albumentations.ai/transform/XYMasking): Masks vertical (time) and horizontal (frequency) stripes

**Domain Adaptation:**
These transforms combine information from multiple images for style transfer-like effects:
- [`A.FDA`](https://explore.albumentations.ai/transform/FDA) (Fourier Domain Adaptation): Swaps low-frequency components between images
- [`A.HistogramMatching`](https://explore.albumentations.ai/transform/HistogramMatching): Modifies histogram to match reference image

**Use Case Example:** If you have abundant data from CT Scanner A but limited data from Scanner B, you can use images from Scanner B as the "style" reference for `FDA` or `HistogramMatching` applied to images from Scanner A.

#### Beyond Albumentations: Batch-Based Augmentations

While Albumentations focuses on per-image transforms, techniques that mix multiple samples within a batch are highly effective for regularization:

- **MixUp:** Linearly interpolates pairs of images and their labels (strongly recommended for classification)
- **CutMix:** Cuts a patch from one image and pastes it onto another; labels are mixed proportionally to the patch area
- **Mosaic:** Combines four images into one larger image (common in object detection like YOLO)
- **CopyPaste:** Copies object instances (using masks) from one image and pastes them onto another

*These require custom dataloader logic or libraries like timm that integrate them.*

#### Usage Guidelines

**Remember to:**
- Visualize effects to ensure they're plausible for your domain
- Start with lower probabilities and magnitudes
- Tune based on validation performance
- Don't use all transforms at once - be selective!

### Step 7: Final Normalization - Standard vs. Sample-Specific

The final step in nearly all pipelines is normalization, typically using [`A.Normalize`](https://explore.albumentations.ai/transform/Normalize). This transform subtracts a mean and divides by a standard deviation (or performs other scaling) for each channel.

*   **Standard Practice (Fixed Mean/Std):** The most common approach is to use pre-computed `mean` and `std` values calculated across a large dataset (like ImageNet). These constants are then applied uniformly to all images during training and inference using the default `normalization="standard"` setting. This standardizes the input distribution for the model.

    ```python
    # Example using typical ImageNet mean/std (default behaviour)
    normalize_fixed = A.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                max_pixel_value=255.0,
                                normalization="standard", # Default
                                p=1.0)
    ```

*   **Sample-Specific Normalization (Built-in):** Albumentations' `A.Normalize` also directly supports calculating the `mean` and `std` *for each individual augmented image* just before normalization, using these statistics to normalize the image. This can act as an additional regularization technique.
    *   *Note:* This technique is highlighted by practitioners like [Christof Henkel](https://www.kaggle.com/christofhenkel) (Kaggle Competitions Grandmaster with more than 40 gold medals from Machine Learning competitions) as a useful regularization method.
    *   **How it works:** When `normalization` is set to `"image"` or `"image_per_channel"`, the transform calculates the statistics from the current image *after* all preceding augmentations have been applied. These dynamic statistics are then used for normalization.
        *   `normalization="image"`: Calculates a single mean and std across all channels and pixels, then normalizes the entire image by these values.
        *   `normalization="image_per_channel"`: Calculates the mean and std independently for each channel, then normalizes each channel by its own statistics.
    *   **Why it helps:**
        *   **Regularization:** This introduces randomness in the final normalized values based on the specific augmentations applied to the sample, acting as a form of data-dependent noise.
        *   **Robustness:** Normalizing by the *current* image's statistics makes the model less sensitive to global shifts in brightness (due to mean subtraction) and contrast/intensity scaling (due to standard deviation division). The model learns to interpret features relative to the image's own statistical properties.
    *   **Implementation:** Simply set the `normalization` parameter accordingly. The `mean`, `std`, and `max_pixel_value` arguments are ignored when using `"image"` or `"image_per_channel"$.

        ```python
        # Example using sample-specific normalization (per channel)
        normalize_sample_per_channel = A.Normalize(normalization="image_per_channel", p=1.0)

        # Example using sample-specific normalization (global)
        normalize_sample_global = A.Normalize(normalization="image", p=1.0)

        # Other options like min-max scaling are also available:
        normalize_min_max = A.Normalize(normalization="min_max", p=1.0)
        ```

Choosing between fixed (`"standard"`) and sample-specific (`"image"`, `"image_per_channel"`) normalization depends on the task and observed performance. Fixed normalization is the standard and usually the starting point, while sample-specific normalization can be experimented with as an advanced regularization strategy.

*   **Min-Max Scaling:** `A.Normalize` also supports `"min_max"` and `"min_max_per_channel"` scaling, which rescale pixel values to the [0, 1] range based on the image's (or channel's) minimum and maximum values. This is another form of sample-dependent normalization.

### Advanced Uses of `OneOf`

While `A.OneOf` is commonly used to select one transform from a list of different transform *types* (e.g., one type of blur, one type of noise), it can also be used in more nuanced ways:

1.  **Creating Distributions over Parameters:** You can use `OneOf` to apply the *same* transform type but with different fixed parameters, effectively creating a custom distribution. For example, to randomly apply either JPEG or WebP compression with a certain quality range:

    ```python
    import albumentations as A

    compression_types = ["jpeg", "webp"]
    compression_variation = A.OneOf([
        A.ImageCompression(quality_range=(20, 80), compression_type=ctype, p=1.0)
        for ctype in compression_types
    ], p=0.5) # Apply one of the compression types 50% of the time
    ```

2.  **Creating Distributions over Methods:** Some transforms have different internal methods for achieving their goal. You can use `OneOf` to randomly select which method is used. For example, to apply grayscale conversion using different algorithms:

    ```python
    import albumentations as A

    grayscale_methods = ["weighted_average", "from_lab", "desaturation", "average", "max", "pca"]
    grayscale_variation = A.OneOf([
        A.ToGray(method=m, p=1.0) for m in grayscale_methods
    ], p=0.3) # Apply one grayscale method 30% of the time
    ```

This allows for finer-grained control over the types of variations introduced by your pipeline.

## Putting It All Together: A Comprehensive (and Potentially Excessive) Example

> **⚠️ WARNING: This is for ILLUSTRATION ONLY**
>
> It is highly unlikely you would use *all* of these transforms simultaneously in a real-world scenario. This example shows how different augmentations can be combined using `A.OneOf`, but **remember the principle: start simple and add complexity incrementally based on validation results!**

Below is an example of a complex pipeline combining many of the discussed techniques:

```python
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2 # If using PyTorch

TARGET_SIZE = 256

# ⚠️ DO NOT USE ALL OF THIS AT ONCE ⚠️
# This is an ILLUSTRATION of how transforms can be combined
heavy_train_pipeline = A.Compose(
    [
        # 1. Initial Resizing/Cropping (Choose ONE strategy)
        A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
        A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
        # Alternative options (choose one):
        # A.RandomResizedCrop(height=TARGET_SIZE, width=TARGET_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
        # A.LongestMaxSize(max_size=TARGET_SIZE, p=1.0), A.PadIfNeeded(...),

        # 2. Basic Geometric (ALWAYS include these first)
        A.HorizontalFlip(p=0.5),
        # A.SquareSymmetry(p=0.5) # Use if appropriate (e.g., aerial)

        # 3. Affine and Perspective (Choose one)
        A.OneOf([
            A.Affine(
                scale=(0.8, 1.2), rotate=(-15, 15),
                translate_percent=(-0.1, 0.1), shear=(-10, 10), p=0.8
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.8)
        ], p=0.7),

        # 4. Dropout / Occlusion (HIGHLY RECOMMENDED)
        A.OneOf([
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                        hole_width_range=(0.1, 0.25), fill_value=0, p=1.0),
            A.GridDropout(ratio=0.5, unit_size_range=(0.05, 0.1), p=0.5),
            A.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ], p=0.5),

        # 5. Color Space Reduction (Use sparingly)
        A.OneOf([
            A.ToGray(p=0.3),
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.3),
        ], p=0.2),

        # 6. Color Augmentations (Choose based on domain)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
            A.RandomGamma(gamma_limit=(80, 120), p=0.8),
        ], p=0.7),

        # 7. Blur (Low probability)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
        ], p=0.3),

        # 8. Noise (Low probability)
        A.OneOf([
            A.GaussNoise(std_limit=(0.1, 0.2), p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
            A.SaltAndPepper(p=0.5)
        ], p=0.3),

        # 9. Compression / Downscaling (Very low probability)
        A.OneOf([
            A.ImageCompression(quality_range=(20, 80), p=0.5),
            A.Downscale(scale_range=(0.25, 0.5), p=0.5),
        ], p=0.2),

        # --- Final Steps ---
        # 10. Normalization (ALWAYS do this last)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                   max_pixel_value=255.0, normalization="standard", p=1.0),

        # 11. Convert to Tensor (if using PyTorch)
        # ToTensorV2(p=1.0),
    ],
    # Add bbox_params or keypoint_params if dealing with bounding boxes or keypoints
    # bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])
    #
    # IMPORTANT: Multi-target compatibility - not all transforms support all targets!
    # Check the Supported Targets by Transform reference to verify compatibility:
    # https://albumentations.ai/docs/reference/supported-targets-by-transform/
)

# ✅ BETTER: Start with this simple pipeline instead
recommended_starter = A.Compose([
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
])
```

**Remember:** Always visualize your pipeline output and measure validation performance after each addition!

## Where to Go Next?

Armed with strategies for choosing augmentations, you can now:

-   **[Apply to Your Specific Task](./):** Integrate your chosen transforms into the pipeline for your task (e.g., Classification, Segmentation, Detection).
-   **[Check Transform Compatibility](../reference/supported-targets-by-transform.md):** Essential when working with multiple targets - verify which transforms support your specific combination of images, masks, bboxes, or keypoints.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** **Upload your own images** and experiment interactively with the specific transforms and parameters you are considering. See real-time results on your actual data.
-   **[Optimize Pipeline Speed](./performance-tuning.md):** Ensure your selected augmentation pipeline is efficient and doesn't bottleneck training.
-   **[Review Core Concepts](../2-core-concepts/index.md):** Reinforce your understanding of how pipelines, probabilities, and targets work with your chosen transforms.
-   **[Dive into Advanced Guides](../4-advanced-guides/index.md):** If standard transforms aren't enough, learn how to create custom ones.
