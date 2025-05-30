# Working with Multiple Data Targets

Albumentations pipelines can apply augmentations consistently across various types of related data, called **targets**.
Beyond just augmenting the input `image`, you can simultaneously augment corresponding masks, bounding boxes, keypoints,
and volumetric data, ensuring spatial transformations are synchronized.

## Quick Reference

**Key Concepts:**
- **`image` or `images`**: Required - primary input data
- **Spatial transforms**: Affect all targets (image, mask, bboxes, keypoints, volumes)
- **Pixel transforms**: Only affect image-like targets (image, images, volumes)
- **Synchronized augmentation**: All targets receive identical spatial transformations
- **Format configuration**: `bbox_params` and `keypoint_params` required for respective targets

**Common Target Combinations:**
- **Classification**: `image`
- **Segmentation**: `image` + `mask`
- **Object Detection**: `image` + `bboxes` + `bbox_params`
- **Keypoint Detection**: `image` + `keypoints` + `keypoint_params`
- **3D Data Processing**: `volume` + `mask3d`

## Core Requirements

### Mandatory Input

**You must provide either an `image` or `images` keyword argument.** Other targets like `mask`, `bboxes`, etc.,
are optional and depend on your specific task. All data is passed as keyword arguments to the pipeline call.

### Data Type Requirements

All image-like data (image, images, volume, volumes) **must be `uint8` or `float32`** NumPy arrays.
Masks can be any integer type, while bboxes and keypoints are typically `float32`.

## 2D Targets

### Single Image Data

#### `image`: Primary Input Image

**Description:** The main input image for augmentation.

**Format:** NumPy array with shape:
- `(height, width, channels)` for color images (e.g., RGB)
- `(height, width)` for grayscale images

**Example:**
```python
import numpy as np
import albumentations as A

# Color image
image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)

# Grayscale image
image_gray = np.random.randint(0, 256, (224, 224), dtype=np.uint8)

transform = A.Compose([A.HorizontalFlip(p=0.5)])
result = transform(image=image)
```

#### `mask`: Segmentation Mask

**Description:** A segmentation mask corresponding to the input image.

**Format:** NumPy array with **same height and width** as the input image:
- `(height, width)` for single-class or multiclass masks
- `(height, width, num_classes)` for multi-channel masks

**Behavior:**
- **Spatial transforms**: Applied identically to image and mask
- **Pixel transforms**: Do not affect masks

**Example:**
```python
import albumentations as A
import numpy as np

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
mask = np.random.randint(0, 5, (100, 100), dtype=np.uint8)  # 5 classes

transform = A.Compose([
    A.HorizontalFlip(p=1.0),           # Affects both image and mask
    A.RandomBrightnessContrast(p=1.0), # Affects only image
])

result = transform(image=image, mask=mask)
transformed_image = result['image']
transformed_mask = result['mask']  # Flipped but not brightness-adjusted
```

### Multiple Image Data

#### `images`: Batch of Images

**Description:** Multiple images that receive **identical** augmentation parameters. Essential for video frames,
stereo pairs, or multi-channel data requiring synchronized transformations.

**Format:** NumPy array with shape:
- `(num_images, height, width, channels)` for color images
- `(num_images, height, width)` for grayscale images

**Key Feature:** All images receive the exact same sequence and parameters of augmentations.

#### `masks`: Multiple Masks

**Description:** Multiple segmentation masks for instance segmentation or multi-object scenarios.

**Format:** `(num_masks, height, width)` - each slice `[i, :, :]` represents one mask.

**Use Case:** Instance segmentation where each mask represents a different object instance.

### Coordinate-Based Targets

#### `bboxes`: Bounding Boxes

**Description:** Object bounding boxes with configurable coordinate formats.

**Requirements:**
- Must specify `bbox_params` in `A.Compose`
- Coordinates and labels handled separately

**Supported Formats:** `pascal_voc`, `albumentations`, `coco`, `yolo`

**Example:**
```python
import albumentations as A
import numpy as np

image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
bboxes = np.array([[10, 10, 50, 50], [60, 60, 90, 90]])  # [x_min, y_min, x_max, y_max]
class_labels = [1, 2]

transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=0.5),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['class_labels']
))

result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
```

#### `keypoints`: Keypoints/Landmarks

**Description:** Points of interest with configurable formats and coordinate systems.

**Requirements:**
- Must specify `keypoint_params` in `A.Compose`
- Support for 2D and 3D coordinates

**Common Formats:** `[x, y]`, `[x, y, angle, scale]`, `[x, y, z]`

**3D Note:** For XYZ keypoints, 2D transforms only modify x, y coordinates; z remains unchanged.

## 3D Targets (Volumetric Data)

### Single Volume Data

#### `volume`: 3D Data Volumes

**Description:** 3D data volumes for various applications including medical imaging (CT, MRI), scientific simulations,
geospatial data, computer graphics, and other volumetric datasets.

**Format:**
- `(depth, height, width, channels)` (DHWC)
- `(depth, height, width)` (DHW)

**Transform Behavior:**
- **2D transforms**: Applied slice-wise with identical parameters across all slices
- **3D transforms**: Applied to the entire volume
- **Pixel transforms**: Applied slice-wise

#### `mask3d`: 3D Segmentation Mask

**Description:** 3D segmentation mask corresponding to a volume, used for labeling regions in 3D space.

**Format:** `(depth, height, width)`

### Multiple Volume Data

#### `volumes`: Batch of 3D Volumes

**Format:** `(num_volumes, depth, height, width, channels)` or `(num_volumes, depth, height, width)`

#### `masks3d`: Multiple 3D Masks

**Format:** `(num_masks, depth, height, width)`

## Target Compatibility Matrix

| Transform Type | image | mask | bboxes | keypoints | volume | mask3d |
|---------------|-------|------|--------|-----------|--------|---------|
| **Spatial** (Flip, Rotate, Crop) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Pixel** (Brightness, Blur, Noise) | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Geometric** (Affine, Perspective) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Practical Examples

### Semantic Segmentation

```python
import albumentations as A
import numpy as np

# Prepare data
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
mask = np.random.randint(0, 21, (256, 256), dtype=np.uint8)  # 21 classes

# Create pipeline
transform = A.Compose([
    A.RandomCrop(width=224, height=224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(),
])

# Apply augmentations
result = transform(image=image, mask=mask)
```

### Object Detection

```python
import albumentations as A
import numpy as np

# Prepare data
image = np.random.randint(0, 256, (416, 416, 3), dtype=np.uint8)
bboxes = np.array([[50, 50, 150, 150], [200, 200, 350, 350]])
labels = ['person', 'car']

# Create pipeline with bbox parameters
transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=320, height=320),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(
    format='pascal_voc',
    label_fields=['labels'],
    min_area=1024,
    min_visibility=0.1
))

# Apply augmentations
result = transform(image=image, bboxes=bboxes, labels=labels)
```

### Multi-Target Pipeline

```python
import albumentations as A
import numpy as np

# Prepare all data types
image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
mask = np.random.randint(0, 2, (300, 300), dtype=np.uint8)
bboxes = np.array([[25, 25, 100, 100], [150, 150, 250, 250]])
keypoints = np.array([[50, 50], [200, 200]])
class_labels = [1, 2]
keypoint_labels = ['nose', 'eye']

# Create comprehensive pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
],
bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']),
keypoint_params=A.KeypointParams(format='xy', label_fields=['keypoint_labels'])
)

# Apply to all targets
result = transform(
    image=image,
    mask=mask,
    bboxes=bboxes,
    keypoints=keypoints,
    class_labels=class_labels,
    keypoint_labels=keypoint_labels
)
```

## Best Practices

### Data Consistency

1. **Shape Matching**: Ensure all spatial targets have matching dimensions
2. **Data Types**: Use `uint8` for images, appropriate types for other targets
3. **Coordinate Systems**: Verify bbox/keypoint formats match your data

### Performance Optimization

1. **Batch Processing**: Use `images`/`masks` for multiple related items
2. **Target Selection**: Only include targets you actually need
3. **Memory Management**: Consider data types and array sizes

### Common Pitfalls

1. **Missing Parameters**: Always provide `bbox_params`/`keypoint_params` when using bboxes/keypoints
2. **Format Mismatches**: Ensure coordinate formats match your data
3. **Shape Inconsistencies**: All targets must have compatible spatial dimensions

## Where to Go Next?

Now that you understand how Albumentations handles different data targets, you can:

**Task-Specific Guides:**
-   **[Semantic Segmentation](../3-basic-usage/semantic-segmentation.md)** - Working with `image` and `mask`
-   **[Object Detection](../3-basic-usage/bounding-boxes-augmentations.md)** - Using `image`, `bboxes`, and `bbox_params`
-   **[Keypoint Detection](../3-basic-usage/keypoint-augmentations.md)** - Handling `image`, `keypoints`, and `keypoint_params`
-   **[Volumetric Data](../3-basic-usage/volumetric-augmentation.md)** - Working with `volume`, `mask3d`, and 3D transforms

**Advanced Topics:**
-   **[Additional Targets](../4-advanced-guides/additional-targets.md)** - Define custom data types beyond standard targets
-   **[Pipelines](./pipelines.md)** - Understand how `A.Compose` orchestrates transforms across targets
-   **[Transform Compatibility](../reference/supported-targets-by-transform.md)** - See which transforms support which targets

**Interactive Learning:**
-   **[Explore Transforms](https://explore.albumentations.ai)** - Visualize how different transforms affect various targets
