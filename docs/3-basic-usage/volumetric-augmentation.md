# Introduction to 3D Medical Image Augmentation

## Overview

While primarily used for medical imaging (CT scans, MRI), Albumentations' 3D transforms can be applied to various volumetric data types

### Medical Imaging
- CT and MRI scans
- Ultrasound volumes
- PET scans
- Multi-modal medical imaging

### Scientific Data
- Microscopy z-stacks
- Cryo-EM volumes
- Geological seismic data
- Weather radar volumes

### Industrial Applications
- 3D NDT (Non-Destructive Testing) scans
- Industrial CT for quality control
- Material analysis volumes
- 3D ultrasonic testing data

### Computer Vision
- Depth camera sequences
- LiDAR point cloud voxelizations
- Multi-view stereo reconstructions

## Data Format

### Volumes
Albumentations expects 3D volumes as numpy arrays in the following formats:
- `(D, H, W)` - Single-channel volumes (e.g., CT scans)
- `(D, H, W, C)` - Multi-channel volumes (e.g., multi-modal MRI)

Where:
- D = Depth (number of slices)
- H = Height
- W = Width
- C = Channels (optional)

### 3D Masks
Segmentation masks should match the volume dimensions:
- `(D, H, W)` - Binary or single-class masks
- `(D, H, W, C)` - Multi-class masks

When applying augmentations, pass these masks using the `mask3d` keyword argument.

## Basic Usage

```python
import albumentations as A
import numpy as np
```

### Create a basic 3D augmentation pipeline

```python
transform = A.Compose([
    # Crop volume to a fixed size for memory efficiency
    A.RandomCrop3D(size=(64, 128, 128), p=1.0),
    # Randomly remove cubic regions to simulate occlusions
    A.CoarseDropout3D(
        num_holes_range=(2, 6),
        hole_depth_range=(0.1, 0.3),
        hole_height_range=(0.1, 0.3),
        hole_width_range=(0.1, 0.3),
        p=0.5
    ),
])
```

### Apply to volume and mask

```python
volume = np.random.rand(96, 256, 256) # Your 3D medical volume
mask = np.zeros((96, 256, 256)) # Your 3D segmentation mask
transformed = transform(volume=volume, mask3d=mask)
transformed_volume = transformed['volume']
transformed_mask = transformed['mask3d']
```
*Note: Geometric transforms like `RandomCrop3D` are automatically applied identically to both the `volume` and the `mask3d` when passed together, ensuring synchronization.*

## Available 3D Transforms

Here are some examples of available 3D transforms:

- [`CenterCrop3D`](https://explore.albumentations.ai/transform/CenterCrop3D) - Crop the center part of a 3D volume
- [`RandomCrop3D`](https://explore.albumentations.ai/transform/RandomCrop3D) - Randomly crop a part of a 3D volume
- [`Pad3D`](https://explore.albumentations.ai/transform/Pad3D) - Pad a 3D volume
- [`PadIfNeeded3D`](https://explore.albumentations.ai/transform/PadIfNeeded3D) - Pad if volume size is less than desired size
- [`CoarseDropout3D`](https://explore.albumentations.ai/transform/CoarseDropout3D) - Random dropout of 3D cubic regions
- [`CubicSymmetry`](https://explore.albumentations.ai/transform/CubicSymmetry) - Apply random cubic symmetry transformations

For a complete and up-to-date list of all available 3D transforms, please see our [API Reference](api-reference/augmentations/3d-transforms.md).

## Combining 2D and 3D Transforms

You can combine 2D and 3D transforms in the same pipeline. When 2D transforms are included, Albumentations samples their random parameters *once* per call and applies them identically to each XY slice along the depth axis. This ensures consistency across slices for transforms like flips or color adjustments.

```python
transform = A.Compose([
    # 3D transforms
    A.RandomCrop3D(size=(64, 128, 128)),
    # 2D transforms (applied identically to each XY slice)
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

transformed = transform(volume=volume, mask3d=mask)
transformed_volume = transformed['volume']
transformed_mask = transformed['mask3d']
```


## Memory Management

3D volumes can be large. Consider using smaller crop sizes or processing in patches.
   - Place cropping operations at the beginning of your pipeline for better performance
   - Example: A `256x256x256` volume cropped to `64x64x64` will process subsequent transforms ~64x faster

### Efficient pipeline - cropping first
```python
efficient_transform = A.Compose([
A.RandomCrop3D(size=(64, 64, 64)), # Do this first!
A.CoarseDropout3D(...),
A.RandomBrightnessContrast(...)
])
```

### Less efficient pipeline - processing full volume unnecessarily
```python
inefficient_transform = A.Compose([
A.CoarseDropout3D(...), # Processing full volume
A.RandomBrightnessContrast(...), # Processing full volume
A.RandomCrop3D(size=(64, 64, 64)) # Cropping at the end
])
```

## Example Pipeline

Here's a complete example of a medical image augmentation pipeline:

```python
import albumentations as A
import numpy as np

def create_3d_pipeline(
    crop_size=(64, 128, 128),
    p_spatial=0.5,
    p_intensity=0.3
    ):
    return A.Compose([
        # Spatial transforms
        A.RandomCrop3D(
            size=crop_size,
            p=1.0
        ),
        A.CubicSymmetry(p=p_spatial),
        # Intensity transforms
        A.CoarseDropout3D(
            num_holes_range=(2, 5),
            hole_depth_range=(0.1, 0.2),
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            p=p_intensity
        ),
    ])
```

### Usage

```python
transform = create_3d_pipeline()
volume = np.random.rand(96, 256, 256)
mask = np.zeros((96, 256, 256))
transformed = transform(volume=volume, mask3d=mask)
```


## Where to Go Next?

After learning the basics of volumetric augmentation:

-   **[Explore 3D Transforms API](../reference/augmentations/3d-transforms.md):** See the full list of available 3D-specific augmentations and their parameters.
-   **[Review Core Concepts](../2-core-concepts):** Understand how [Targets](../2-core-concepts/targets.md) (`volume`, `mask3d`) and [Pipelines](../2-core-concepts/pipelines.md) handle 3D data and mixed 2D/3D transforms.
-   **[Refine Augmentation Choices](./choosing-augmentations.md):** Consider which 2D (slice-wise) and 3D transforms best suit your specific volumetric data and task.
-   **[Optimize Performance](./performance-tuning.md):** Apply strategies to efficiently process large 3D volumes, especially the 'crop early' technique.
-   **Explore Related Task Guides:**
    -   [Video Augmentation](./video-augmentation.md) (For sequences of 2D frames)
    -   [Semantic Segmentation](./semantic-segmentation.md) (For 2D segmentation concepts)
-   **[Dive into Advanced Guides](../4-advanced-guides/index.md):** Learn about creating custom transforms (potentially 3D) or serialization.
-   **[Visually Explore 2D Transforms](https://explore.albumentations.ai):** Experiment with the 2D transforms that can be applied slice-wise to your volumes.
