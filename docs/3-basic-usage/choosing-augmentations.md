# Choosing Augmentations for Model Generalization

Selecting the right set of augmentations is key to helping your model generalize better to unseen data. While applying augmentations, it's also crucial to ensure your pipeline runs efficiently.

**Before diving into *which* augmentations to choose, we strongly recommend reviewing the guide on [Optimizing Augmentation Pipelines for Speed](./performance-tuning.md) to avoid CPU bottlenecks during training.**

This guide focuses on strategies for selecting augmentations that are *useful* for improving model performance.

## 1. Understand Your Task and Data Domain

The most effective augmentations often mimic the variations your model will encounter in the real world or are relevant to the specific computer vision task.

*   **Consider Plausible Variations:** Think about the natural variations in your data domain.
    *   **Lighting:** If lighting conditions vary (time of day, indoor/outdoor), use `RandomBrightnessContrast`, `RandomGamma`.
    *   **Viewpoint/Scale:** If objects appear at different distances or angles, use `RandomResizedCrop`, `Affine` (for scale/translation/rotation), `Perspective`.
    *   **Color:** If color shifts are possible (different cameras, white balance), use `HueSaturationValue`, `ColorJitter`, `RandomBrightnessContrast`.
    *   **Orientation:** If objects can appear flipped or rotated, use `HorizontalFlip`, `VerticalFlip`, `Rotate`, `SquareSymmetry`.
    *   **Occlusion:** If parts of objects might be hidden, consider `CoarseDropout` (Cutout).
    *   **Sensor Noise/Blur:** `GaussNoise`, `GaussianBlur`, `MotionBlur` can simulate sensor imperfections or movement.
    *   **Elastic Deformations:** Particularly relevant for medical imaging or deformable objects (`ElasticTransform`).
*   **Task Requirements:** How does the task influence choices?
    *   **Classification:** Generally robust to geometric and color changes. Ensure augmentations don't remove the object of interest entirely.
    *   **Object Detection / Segmentation / Keypoints:** Geometric augmentations must be applied consistently to the image and its corresponding targets (masks, bboxes, keypoints). Configure `bbox_params` or `keypoint_params` correctly in `A.Compose`. Pixel-level transforms are usually safe as they don't affect coordinates.

## 2. Start Simple and Iterate

Avoid applying every possible augmentation initially. Establish a baseline and build complexity incrementally.

*   **Baseline:** Train with minimal or no augmentations (perhaps just `HorizontalFlip` if relevant) to gauge initial performance.
*   **Add Gradually:** Introduce categories of augmentations (e.g., basic geometric flips/rotates, then color shifts, then noise/blur). Monitor validation metrics to see if additions help or hurt.
*   **Tune Parameters:** Adjust the probability (`p`) and magnitude (e.g., `brightness_limit`, `rotation_limit`) of each transform. Sometimes less intense or less frequent augmentations are more effective.

## 3. Visualize Your Augmentations

**This is a critical step!** Always inspect the output of your pipeline visually on sample data.

*   **Check Realism:** Are the augmented images plausible within your domain? Avoid overly distorted or unrealistic results.
*   **Check Target Consistency:** For detection, segmentation, or keypoints, verify that masks, bounding boxes, and keypoints are correctly transformed along with the image.
*   **Use Tools:** Libraries like `matplotlib`, `cv2`, or tools like [Explore Albumentations](https://explore.albumentations.ai/) can help visualize batches of augmented data.

```python
# Example visualization (conceptual)
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Assume 'transform' is your defined Albumentations pipeline
# Assume 'load_sample_data' loads an image and potentially targets

def visualize_augmentations(pipeline, num_samples=3):
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        # Load a fresh sample each time if possible
        sample_data = load_sample_data() # Replace with your data loading
        original_image = sample_data.get('image')

        augmented_data = pipeline(**sample_data)
        augmented_image = augmented_data['image']

        # --- Plot Original ---
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) if len(original_image.shape) == 3 and original_image.shape[2] == 3 else original_image, cmap='gray')
        plt.title(f"Original Sample {i+1}")
        # Add plotting for original targets (bboxes, masks) if applicable

        # --- Plot Augmented ---
        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB) if len(augmented_image.shape) == 3 and augmented_image.shape[2] == 3 else augmented_image, cmap='gray')
        plt.title(f"Augmented Sample {i+1}")
        # Add plotting for augmented targets (bboxes, masks) if applicable
        # E.g., draw transformed bboxes on augmented_image

    plt.tight_layout()
    plt.show()

# --- Define your pipeline ---
# Example pipeline (replace with yours)
pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']) if False else None) # Add bbox_params if needed

# --- Dummy data loading function (replace with yours) ---
def load_sample_data():
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    # Add dummy bboxes, masks etc. if your pipeline uses them
    # bboxes = np.array([[10, 10, 50, 50]]); class_labels=[1]
    return {'image': img}#, 'bboxes': bboxes, 'class_labels': class_labels}

# --- Visualize ---
# visualize_augmentations(pipeline, num_samples=3)
```

## Conclusion

Choosing effective augmentations is an iterative process involving understanding your data, starting simple, visualizing results, and measuring impact on model performance. Combine these principles with the performance optimizations discussed in the linked guide for a robust and efficient training workflow.
