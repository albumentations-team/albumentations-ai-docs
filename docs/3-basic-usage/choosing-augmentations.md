# How to Pick Augmentations

Choosing the right set of augmentations is crucial for training robust computer vision models. Applying too few or inappropriate augmentations might not improve generalization enough, while applying too many or overly aggressive ones can sometimes harm performance by making the task too difficult for the model.

Here's a breakdown of factors to consider and strategies to employ:

## 1. Understand Your Task and Data

The most important factor is the nature of your specific task and dataset.

*   **Task Type:**
    *   **Classification:** Often benefits from geometric (flips, rotates, scale, translate, shear), color (brightness, contrast, saturation, hue), and noise/blur augmentations. Be cautious with transforms that drastically change object identity (e.g., extreme crops, Cutout if it removes the whole object).
    *   **Object Detection:** Needs augmentations that preserve bounding box accuracy. Geometric transforms are common, but ensure `bbox_params` are correctly configured. Color changes are usually safe. Avoid transforms that might remove objects entirely or significantly change their relative positions if spatial relationships are important.
    *   **Segmentation:** Similar to detection, requires consistency between image and mask transformations. Geometric and color augmentations are widely used. Elastic transformations can be particularly effective. Ensure mask values remain correct (e.g., don't interpolate mask labels during rotation).
    *   **Keypoint Detection:** Needs transforms that handle keypoint coordinates correctly. Geometric transforms require `keypoint_params`. Some noise/blur might be acceptable.
*   **Data Domain:** Consider what variations are naturally present or plausible in your deployment environment.
    *   **Medical Images:** Rotations, flips, elastic deformations, brightness/contrast adjustments are common. Avoid unrealistic color shifts unless relevant (e.g., staining variations).
    *   **Satellite/Aerial Imagery:** Rotations, flips, scale, brightness/contrast are standard. Color shifts might be relevant depending on atmospheric conditions or sensor types.
    *   **Natural Scenes:** A wide range of geometric and color augmentations often applies.
    *   **Documents:** Slight rotations, shear, perspective transforms, brightness/contrast, blur might be useful.
*   **Object Invariance:** Should your model be invariant to certain changes?
    *   **Rotation Invariance:** Use `Rotate` or `RandomRotate90`.
    *   **Scale Invariance:** Use `RandomScale`, `Resize`, `RandomSizedCrop`.
    *   **Lighting Invariance:** Use `RandomBrightnessContrast`, `RandomGamma`.
    *   **Color Invariance:** Use `HueSaturationValue`, `ToGray`.

## 2. Start Simple, Then Iterate

Don't start by throwing every possible augmentation at your model.

*   **Baseline:** Train a model with no or very minimal augmentations (e.g., just HorizontalFlip) to establish a baseline performance.
*   **Add Gradually:** Introduce augmentations one category at a time (e.g., add basic geometric, then color, then noise/blur). Monitor validation performance after adding each group.
*   **Tune Probabilities and Magnitudes:** Once you have a set of candidate augmentations, experiment with their `p` probabilities and the intensity/range of their parameters (e.g., rotation limit, brightness limit). Less frequent or less intense augmentations might be better than aggressive ones.

## 3. Use Domain Knowledge (or Mimic Natural Variations)

Think about how real-world data for your task varies.

*   If your camera might be slightly tilted, add `Rotate`.
*   If lighting conditions vary, add `RandomBrightnessContrast`.
*   If objects appear at different scales, add scaling augmentations.

## 4. Visualize Your Augmentations

**Always** look at the output of your augmentation pipeline on a sample of your images. This is crucial to ensure:

*   The augmentations are realistic and relevant to your domain.
*   They are not overly distorting the images or making objects unrecognizable.
*   Bounding boxes, masks, or keypoints are still correctly aligned after augmentation.
*   You can use libraries like `matplotlib` or dedicated tools (like [Explore Albumentations](https://explore.albumentations.ai/)) to visualize batches of augmented data.

```python
# Example visualization (conceptual)
import albumentations as A
import cv2
import matplotlib.pyplot as plt
import random

# Assume 'transform' is your defined Albumentations pipeline
# Assume 'dataset' is your data source

image, label = random.choice(dataset) # Get a sample

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original")

augmented_data = transform(image=image)
augmented_image = augmented_data['image']

plt.subplot(1, 3, 2)
plt.imshow(augmented_image)
plt.title("Augmented 1")

augmented_data_2 = transform(image=image)
augmented_image_2 = augmented_data_2['image']

plt.subplot(1, 3, 3)
plt.imshow(augmented_image_2)
plt.title("Augmented 2")

plt.show()
```

## 5. Consider Task-Specific Augmentations

Some augmentations are designed for specific challenges:

*   **Cutout / CoarseDropout:** Helps models focus on different parts of an object, useful against occlusion.
*   **MixUp / CutMix:** (Often implemented outside Albumentations, but conceptually related) Mixes images and labels, encouraging linear behavior between classes.
*   **ElasticTransform:** Simulates tissue deformations, often used in medical imaging.

## 6. Leverage Pre-built Policies (Use with Caution)

Techniques like AutoAugment, RandAugment, and TrivialAugment learn augmentation policies automatically. While powerful, they might not always be optimal for your specific dataset compared to a hand-tuned policy. Albumentations doesn't directly implement these learning strategies, but you can often replicate the *resulting* policies using Albumentations transforms.

## Conclusion

Choosing augmentations is an empirical process. Start with domain knowledge, add augmentations incrementally, visualize the results, and monitor validation performance. There's often no single "best" set, and experimentation is key.
