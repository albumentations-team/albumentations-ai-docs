# Video Augmentation with Albumentations

Albumentations can apply augmentations consistently across video frames, which is essential for maintaining temporal coherence in tasks like video object detection or pose estimation. This guide focuses on the mechanics of applying transforms to video frames and associated targets like bounding boxes and keypoints.

## Core Concept: Consistent Augmentation via `images` Target

The key is to treat your video clip as a sequence of images. Load your video frames into a NumPy array with the shape `(N, H, W, C)` (for color) or `(N, H, W)` (for grayscale), where `N` is the number of frames.

When you pass this array to the `images` (plural) argument of your `A.Compose` pipeline, Albumentations understands that the first dimension represents the sequence. It will:

1.  Sample random parameters for each augmentation *once* per call.
2.  Apply the augmentation with those *identical parameters* to every frame (`image`) along the first dimension.

This ensures spatial augmentations (like crops, flips, rotations) and color augmentations are consistent across the clip.

## Basic Workflow (Image Frames Only)

### 1. Setup

```python
import albumentations as A
import cv2
import numpy as np
```

### 2. Define Pipeline

Define a standard pipeline. No special parameters are needed just for video frames themselves.

```python
transform_video_frames = A.Compose([
    A.Resize(height=256, width=256),
    A.RandomCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.2),
    # Add Normalize and ToTensorV2 if needed for your model
    # A.Normalize(...),
    # A.ToTensorV2(),
])
```

### 3. Load Video Frames

Load your video into the expected NumPy format.

```python
# Example: 10 RGB frames of size 360x480
num_frames = 10
height, width = 360, 480
video = np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)
```

### 4. Apply Transform

Pass the video array to the `images` argument.

```python
augmented = transform_video_frames(images=video)
augmented_video_frames = augmented['images']

print(f"Original shape: {video.shape}")
print(f"Augmented shape: {augmented_video_frames.shape}")
# Example output (shape depends on pipeline): Augmented shape: (10, 224, 224, 3)
```

The same crop, flip, and color jitter parameters were applied to all 10 frames.

```python
# Conceptual Example: Applying transforms to video frames and masks

# 1. Prepare Data
num_frames = 10
height, width = 360, 480
video = np.random.randint(0, 256, (num_frames, height, width, 3), dtype=np.uint8)
# Example: Binary masks (0 or 1) for each frame
video_masks = np.random.randint(0, 2, (num_frames, height, width), dtype=np.uint8)

# 2. Define Pipeline
transform_video_seg = A.Compose([
    A.Resize(height=256, width=256),
    A.RandomCrop(height=224, width=224),
    A.HorizontalFlip(p=0.5),
    # Only image gets color jitter
    A.ColorJitter(p=0.2),
    # Normalize image only
    # A.Normalize(...),
    # ToTensorV2 converts both image and mask
    # A.ToTensorV2(),
])

# 3. Apply Transform
augmented = transform_video_seg(images=video, masks=video_masks) # Use 'masks' (plural)
augmented_video_frames = augmented['images']
augmented_video_masks = augmented['masks'] # Extract 'masks' (plural)

print(f"Original Video Shape: {video.shape}")
print(f"Original Masks Shape: {video_masks.shape}")
print(f"Augmented Video Shape: {augmented_video_frames.shape}")
print(f"Augmented Masks Shape: {augmented_video_masks.shape}")
```

## Handling Bounding Boxes and Keypoints for Video

Synchronizing bounding boxes or keypoints across video frames requires careful handling of how targets are passed alongside the `images` array.

**Important Note:** The standard examples for bounding boxes and keypoints typically show passing `image` (singular) and `bboxes`/`keypoints` (plural, for that single image). Applying this directly to video frames passed as `images` (plural) needs consideration of how per-frame targets are structured and processed.

Here are potential approaches:

### Approach 1: Flatten Targets + `frame_id` in `label_fields`

This approach flattens all bounding boxes (or keypoints) from all frames into a single list and uses `label_fields` to track the original frame index for each target.

1.  **Prepare Data:**
    *   Create a single flat list `all_bboxes` containing all boxes from all frames.
    *   Create a corresponding list `frame_indices` indicating the frame index (0 to N-1) for each box in `all_bboxes`.
    *   Create a list `class_labels` for the actual class of each box in `all_bboxes`.

    ```python
    # Conceptual Example Data Preparation (BBoxes)
    # Assume video has 2 frames (N=2)
    # Frame 0 boxes: [[10, 10, 50, 50], [70, 70, 90, 90]] Labels: ['cat', 'dog']
    # Frame 1 boxes: [[20, 20, 60, 60]] Label: ['cat']

    all_bboxes = np.array([[10, 10, 50, 50], [70, 70, 90, 90], [20, 20, 60, 60]])
    frame_indices = [0, 0, 1] # Frame ID for each box
    class_labels = ['cat', 'dog', 'cat'] # Actual class label for each box

    # Conceptual Example Data Preparation (Keypoints)
    # Assume video has 2 frames (N=2)
    # Frame 0 keypoints: [(15, 15), (25, 25)] Labels: ['eye', 'nose']
    # Frame 1 keypoints: [(35, 35)] Label: ['eye']

    # Prepare as NumPy array (N_kp, 2) for 'xy' format
    all_keypoints = np.array([
        [15, 15],
        [25, 25],
        [35, 35]
    ], dtype=np.float32)
    kp_frame_indices = [0, 0, 1]
    kp_class_labels = ['eye', 'nose', 'eye']
    ```

2.  **Define Pipeline:** Include `frame_indices` (or similar) in `label_fields` for `bbox_params` or `keypoint_params`.

    ```python
    transform_video_bboxes = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        # ... other transforms ...
    ], bbox_params=A.BboxParams(format='pascal_voc',
                               label_fields=['class_labels', 'frame_indices']))

    transform_video_keypoints = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        # ... other transforms ...
    ], keypoint_params=A.KeypointParams(format='xy',
                                       label_fields=['kp_class_labels', 'kp_frame_indices']))
    ```

3.  **Apply Transform:** Pass the video frames, the flattened targets, and the corresponding label lists.

    ```python
    # For BBoxes
    augmented_bbox_data = transform_video_bboxes(
        images=video,
        bboxes=all_bboxes,
        class_labels=class_labels,
        frame_indices=frame_indices
    )
    augmented_bboxes = augmented_bbox_data['bboxes']
    augmented_labels = augmented_bbox_data['class_labels']
    augmented_frame_indices = augmented_bbox_data['frame_indices']

    # For Keypoints
    augmented_kp_data = transform_video_keypoints(
        images=video,
        keypoints=all_keypoints,
        kp_class_labels=kp_class_labels,
        kp_frame_indices=kp_frame_indices
    )
    augmented_keypoints = augmented_kp_data['keypoints']
    augmented_kp_labels = augmented_kp_data['kp_class_labels']
    augmented_kp_frame_indices = augmented_kp_data['kp_frame_indices']
    ```

4.  **Post-processing (Regrouping):** You **must** regroup the augmented targets based on the frame indices to reconstruct the per-frame annotations. This can be done efficiently using NumPy indexing.

    ```python
    # Example regrouping for bounding boxes using NumPy
    num_augmented_frames = augmented_bbox_data['images'].shape[0]

    # Ensure label/index lists are NumPy arrays for efficient indexing
    # augmented_bboxes is already a NumPy array if input was NumPy
    augmented_labels_np = np.array(augmented_labels) # Adjust dtype if necessary
    augmented_frame_indices_np = np.array(augmented_frame_indices, dtype=int)

    bboxes_by_frame = []
    for i in range(num_augmented_frames):
        frame_mask = (augmented_frame_indices_np == i)
        bboxes_for_frame = augmented_bboxes[frame_mask] # Index directly
        labels_for_frame = augmented_labels_np[frame_mask]
        bboxes_by_frame.append(bboxes_for_frame) # List of NumPy arrays

    # Now bboxes_by_frame[i] contains a NumPy array of boxes for augmented frame i
    # Same logic applies for regrouping keypoints using kp_frame_indices
    ```
This approach ensures consistent geometric transformations but requires careful pre- and post-processing. Transforms involving cropping or visibility checks (`RandomCrop`, `min_visibility`, `min_area`) might behave unexpectedly on the flattened list; verify their behavior for your use case.


### Approach 2: Using `xyz` format for Keypoints

For keypoints only, one can encode the frame index as the `z` coordinate using the `xyz` format.

1.  **Prepare Data:** Create keypoints where the third element is the frame index.

    ```python
    # Conceptual Example Data Preparation (Keypoints)
    # Frame 0 keypoints: [(15, 15, 0), (25, 25, 0)] Labels: ['eye', 'nose']
    # Frame 1 keypoints: [(35, 35, 1)] Label: ['eye']

    # Prepare as NumPy array (N_kp, 3) for 'xyz' format
    all_keypoints_xyz = np.array([
        [15, 15, 0],
        [25, 25, 0],
        [35, 35, 1]
    ], dtype=np.float32)
    kp_class_labels = ['eye', 'nose', 'eye']
    ```

2.  **Define Pipeline:** Use `format='xyz'`.

    ```python
    transform_video_keypoints_xyz = A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # ... other 2D transforms ...
    ], keypoint_params=A.KeypointParams(format='xyz',
                                       label_fields=['kp_class_labels']))
    ```

3.  **Apply Transform:**

    ```python
    augmented_kp_data_xyz = transform_video_keypoints_xyz(
        images=video,
        keypoints=all_keypoints_xyz,
        kp_class_labels=kp_class_labels
    )
    augmented_keypoints_xyz = augmented_kp_data_xyz['keypoints']
    augmented_kp_labels = augmented_kp_data_xyz['kp_class_labels']
    ```

4.  **Post-processing:** Extract the frame index from the third element (`z`) of each augmented keypoint and regroup.

    ```python
    # Example regrouping for XYZ keypoints using NumPy
    num_augmented_frames = augmented_kp_data_xyz['images'].shape[0]

    # Ensure label list is NumPy array for efficient indexing
    # augmented_keypoints_xyz is already a NumPy array if input was NumPy
    augmented_kp_labels_np = np.array(augmented_kp_labels) # Adjust dtype if necessary

    keypoints_by_frame_xyz = [[] for _ in range(num_augmented_frames)]
    labels_by_frame_xyz = [[] for _ in range(num_augmented_frames)]

    # Extract frame index (z-coordinate) and x, y
    frame_indices_float = augmented_keypoints_xyz[:, 2]
    frame_indices_int = np.round(frame_indices_float).astype(int)
    xy_coords = augmented_keypoints_xyz[:, :2]

    for i in range(num_augmented_frames):
        frame_mask = (frame_indices_int == i)
        keypoints_for_frame = xy_coords[frame_mask]
        labels_for_frame = augmented_kp_labels_np[frame_mask]
        keypoints_by_frame_xyz[i] = keypoints_for_frame # Assign array directly
        labels_by_frame_xyz[i] = labels_for_frame # Assign array directly

    # Now keypoints_by_frame_xyz[i] contains a NumPy array of (x,y) keypoints for frame i
    ```

## Where to Go Next?

After learning how to augment video frames and associated targets:

-   **[Review Core Concepts](../2-core-concepts):** Understand how [Targets](../2-core-concepts/targets.md) (especially `images`) and [Pipelines](../2-core-concepts/pipelines.md) work fundamentally.
-   **[Refine Your Augmentation Choices](./choosing-augmentations.md):** Consider which transforms are most suitable for video data while maintaining temporal consistency.
-   **[Optimize Performance](./performance-tuning.md):** Learn how to speed up your pipeline, which is crucial for potentially large video datasets.
-   **Explore Related Task Guides:** See how targets are handled in other contexts:
    -   [Bounding Box Augmentation](./bounding-boxes-augmentations.md)
    -   [Keypoint Augmentation](./keypoint-augmentations.md)
    -   [Semantic Segmentation](./semantic-segmentation.md)
    -   [Volumetric Augmentation](./volumetric-augmentation.md) (For true 3D data)
-   **[Dive into Advanced Guides](../4-advanced-guides):** Explore custom transforms or serialization if needed for complex video workflows.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Experiment with transforms to see their effect, keeping temporal consistency in mind.
