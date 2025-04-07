# Working with Multiple Data Targets

Albumentations pipelines, defined using `A.Compose`, are designed to apply augmentations consistently across various types of related data, often called **targets**. Beyond just augmenting the input `image`, you can simultaneously augment corresponding masks, bounding boxes, keypoints, and even volumetric data, ensuring spatial transformations are synchronized.

When defining and calling a pipeline, **you must provide either an `image` or `images` keyword argument.** Other targets like `mask`, `bboxes`, etc., are optional and depend on your specific task.

You pass these different data types as keyword arguments to the pipeline call.

## Standard Target Keywords

Albumentations recognizes specific keywords for different data types. Here's a breakdown:

### `image`: Single Image

*   **Description:** The primary input image.
*   **Expected Format:** A NumPy array, typically with shape `(height, width, channels)` (e.g., HWC for RGB) or `(height, width)` (HW for grayscale). **Must be `uint8` or `float32`.**

### `images`: Multiple Images

*   **Description:** Multiple images provided as a single batch. Albumentations ensures that *all* images in this batch receive the *exact same* sequence and parameters of augmentations determined by one call to the pipeline. This is crucial for tasks requiring consistent transformations across related images, such as processing video frames, stereo pairs, or multi-channel satellite imagery where the spatial relationship and applied changes must be identical.
*   **Expected Format:** A NumPy array with shape `(num_images, height, width, channels)` or `(num_images, height, width)`. **Must be `uint8` or `float32`.**

### `mask`: Single Mask

*   **Description:** A segmentation mask corresponding to the `image`.
*   **Expected Format:** A NumPy array with the **same height and width** as the input `image`. Typically has shape `(height, width)` (for binary or multiclass masks where each pixel value is a class ID) or `(height, width, num_classes)` (for multi-channel masks).
*   **Handling:** Pixel-level transforms generally don't affect masks. Spatial transforms (like flips, rotations, resizing) are applied identically to both the `image` and the `mask`.

### `masks`: Multiple Masks

*   **Description:** Multiple segmentation masks provided as a single batch, corresponding to the `image`. Similar to `images`, Albumentations ensures that *all* masks in this batch receive the *exact same* sequence and parameters of spatial augmentations determined by one call to the pipeline. This is often used in tasks like **instance segmentation**, where each mask channel represents a distinct object instance.
*   **Expected Format:** A NumPy array with shape `(num_masks, height, width)` and sharing the same height/width as the input `image`. Each channel `[i, :, :]` represents one mask.
*   **Handling:** Similar to `mask`, spatial transforms are synchronized across the image and all masks.

### `bboxes`: Bounding Boxes

*   **Description:** Bounding boxes associated with objects in the `image`.
*   **Expected Format:** A NumPy array, where each row represents one bounding box. Albumentations expects bounding boxes coordinates and their class labels to be provided.
*   **Handling:** Pixel-level transforms do *not* affect bounding boxes. Spatial transforms modify the coordinates of the boxes to match the image geometry changes. You *must* configure the format and specify how class labels are associated using `bbox_params` in `A.Compose`.
*   **Supported Formats:** Albumentations supports multiple coordinate formats via the `format` argument in `A.BboxParams`, including `pascal_voc`, `albumentations`, `coco`, and `yolo`. Class labels are typically passed as a separate argument to the pipeline call, specified via `label_fields` in `A.BboxParams`. See [Bounding Box Augmentation Details](../3-basic-usage/bounding-boxes-augmentations.md) for more information.

### `keypoints`: Keypoints

*   **Description:** Keypoints or landmarks associated with objects or structures in the `image`.
*   **Expected Format:** A NumPy array, where each row represents one keypoint. Common formats include `[x, y]`, `[x, y, angle, scale]`, or even 3D keypoints `[x, y, z]`. Class labels or other metadata are often associated separately.
*   **Handling:** Similar to `bboxes`, pixel-level transforms don't affect keypoints. Spatial transforms modify their coordinates. You *must* configure the format and specify how any associated labels are handled using `keypoint_params` in `A.Compose`.
    *   **Note on 3D Keypoints:** If you provide keypoints in XYZ format, standard 2D spatial augmentations (like `HorizontalFlip`, `Rotate`, etc.) will only modify the `x` and `y` coordinates, leaving the `z` coordinate unchanged. You would need specific 3D transforms to affect the `z` coordinate.
*   **Details:** See [Keypoint Augmentation Details](../3-basic-usage/keypoint-augmentations.md) for more information on formats and configuration.

### `volume`: Single 3D Volume

*   **Description:** A 3D data volume, often used in medical imaging (e.g., CT or MRI scans).
*   **Expected Format:** A NumPy array, typically with shape `(depth, height, width, channels)` (DHWC) or `(depth, height, width)` (DHW). **Must be `uint8` or `float32`.**
*   **Handling:**
    *   You can apply both specific 3D augmentations and standard 2D augmentations to volumes.
    *   When applying a **2D spatial transform** (like `HorizontalFlip`, `Rotate`, `Affine`), Albumentations treats the volume as a batch of 2D slices along the depth axis. The *exact same* randomly generated parameters for the 2D transform are applied identically to *every* XY slice in the volume.
    *   Pixel-level 2D transforms are also applied slice-wise.
    *   Specific **3D transforms** operate on the volume as a whole.
*   **Details:** See [Volumetric Augmentation Details](../3-basic-usage/volumetric-augmentation.md) for more information on 3D transforms and handling.

### `volumes`: Multiple 3D Volumes

*   **Description:** Multiple 3D volumes provided as a single batch, analogous to `images` for 2D.
*   **Expected Format:** A NumPy array with shape `(num_volumes, depth, height, width, channels)` or `(num_volumes, depth, height, width)`. **Must be `uint8` or `float32`.**
*   **Handling:**
    *   Similar to `volume`, both 2D and 3D transforms can be applied.
    *   When applying a **2D transform**, it operates slice-wise. The *exact same* random parameters are determined once and applied identically to *all corresponding slices* across *all volumes* in the batch (e.g., slice `[:, d, :, :]` across all volumes gets the same 2D augmentation).
    *   Specific **3D transforms** operate on each volume in the batch.
*   **Details:** See [Volumetric Augmentation Details](../3-basic-usage/volumetric-augmentation.md) for more information.

### `mask3d`: Single 3D Mask

*   **Description:** A 3D segmentation mask corresponding to a `volume`.
*   **Expected Format:** A NumPy array, typically `(depth, height, width)`.
*   **Handling:** Similar to 2D masks, spatial 3D transforms are applied.
*   **Details:** See [Volumetric Augmentation Details](../3-basic-usage/volumetric-augmentation.md) for information on usage with 3D pipelines.

### `masks3d`: Multiple 3D Masks

*   **Description:** Multiple 3D segmentation masks provided as a single batch, corresponding to a `volume`.
*   **Expected Format:** A NumPy array with shape `(num_masks, depth, height, width)`.
*   **Details:** See [Volumetric Augmentation Details](../3-basic-usage/volumetric-augmentation.md) for information on usage with 3D pipelines.

## Example: Applying a Pipeline to Image, Mask, and BBoxes

```python
import albumentations as A
import numpy as np

# Dummy Data
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
mask = np.random.randint(0, 2, (100, 100), dtype=np.uint8) # Binary mask
# Bounding boxes (coordinates only) as NumPy array
bboxes = np.array([
    [10, 10, 50, 50], # [x_min, y_min, x_max, y_max]
    [60, 60, 90, 90]
])
class_labels = [1, 2] # Separate list for class labels

# Define Pipeline with Bbox Params
pipeline = A.Compose([
    A.HorizontalFlip(p=1.0), # Spatial transform affects image, mask, bboxes
    A.RandomBrightnessContrast(p=0.5) # Pixel transform affects only image
], bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['class_labels'] # Tells Albumentations labels are in this argument
    )
)

# Pass data as keyword arguments
# Note 'class_labels' matches the entry in `label_fields`
transformed = pipeline(
    image=image,
    mask=mask,
    bboxes=bboxes,
    class_labels=class_labels
)

# Access transformed data
transformed_image = transformed['image']
transformed_mask = transformed['mask']
transformed_bboxes = transformed['bboxes']

print("Original BBoxes:\n", bboxes)
print("Transformed BBoxes:\n", transformed_bboxes)
print("Class Labels (unchanged):", transformed['class_labels'])
```

**Key Points:**

*   Data types are passed using their designated keyword arguments (`image=`, `mask=`, `bboxes=`, etc.).
*   For `bboxes` and `keypoints`, you **must** define `bbox_params` and/or `keypoint_params` in `A.Compose` to specify the format and any associated label fields.
*   Spatial transforms automatically synchronize coordinates across compatible targets (image, mask(s), bboxes, keypoints, volume(s), mask3d(s)).
*   Pixel-level transforms typically only modify the `image` (and potentially `images` or `volume(s)`).
*   Consult the documentation for specific transforms to see exactly which targets they support.

## Where to Go Next?

Now that you know how Albumentations handles different data targets, you can:

-   **See Task-Specific Examples:**
    -   [Semantic Segmentation](../3-basic-usage/semantic-segmentation.md) (using `image` and `mask`)
    -   [Bounding Box Augmentation](../3-basic-usage/bounding-boxes-augmentations.md) (using `image`, `bboxes`, and `bbox_params`)
    -   [Keypoint Augmentation](../3-basic-usage/keypoint-augmentations.md) (using `image`, `keypoints`, and `keypoint_params`)
    -   [Volumetric Augmentation](../3-basic-usage/volumetric-augmentation.md) (using `volume`, `mask3d`, etc.)
-   **Learn About [Additional Targets](../4-advanced-guides/additional-targets.md):** Define how to handle custom data types beyond the standard ones.
-   **Review [Pipelines](./pipelines.md):** Understand how `A.Compose` orchestrates the application of transforms across all provided targets.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** See which augmentations are spatial (affecting targets) and which are pixel-level.
