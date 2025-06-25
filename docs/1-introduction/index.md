# Getting Started with Albumentations

> **Note:** Looking for the actively maintained version? Check out **[AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX)** - a 100% drop-in replacement with improved performance, bug fixes, and new features. Learn more in our [Installation Guide](./installation.md).

Welcome to Albumentations! If you're looking to boost your computer vision model's performance and robustness, you're in the right place. Albumentations is a powerful, flexible, and fast library for image augmentation.

## What is Image Augmentation?

Image augmentation is a technique used to artificially expand the size of a training dataset by creating modified versions of its images. By applying various transformations like rotations, flips, brightness adjustments, or adding noise, you expose your model to a wider variety of data scenarios. This helps prevent overfitting and improves the model's ability to generalize to new, unseen data.

## Why Albumentations?

-   **Fast:** Albumentations is optimized for speed, leveraging highly optimized libraries like OpenCV and NumPy under the hood. It's often significantly faster than other augmentation libraries.
-   **Versatile:** It supports a wide range of computer vision tasks, including classification, segmentation, object detection, and keypoint estimation, handling not just images but also corresponding masks, bounding boxes, and keypoints.
-   **Comprehensive:** Offers a vast collection of diverse augmentation transformations, from simple flips to complex domain-specific effects.
-   **Easy to Use:** Provides a clear and concise API for defining and executing complex augmentation pipelines.
-   **Framework Agnostic:** Integrates seamlessly with popular deep learning frameworks like PyTorch and TensorFlow/Keras.

## Core Concepts

As you explore Albumentations, you'll encounter these key ideas:

1.  **Transforms:** Individual augmentation operations (e.g., [`Rotate`](https://explore.albumentations.ai/transform/Rotate), [`GaussianBlur`](https://explore.albumentations.ai/transform/GaussianBlur), [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast)). Each transform typically has parameters to control its behavior and a probability `p` to control how often it's applied.
2.  **Pipelines (`Compose`):** Chains of transforms are defined using `Compose`. This allows you to sequence multiple augmentations and apply them together. `Compose` also has its own probability `p`.
3.  **Targets:** Albumentations applies transformations consistently across different types of data associated with an image, such as masks, bounding boxes, and keypoints. You specify what you're passing in (e.g., `image`, `mask`, `bboxes`).

## Where to Go Next?

Ready to dive in? Here are some recommended next steps:

-   **[Installation](./installation.md):** Get Albumentations set up in your environment.
-   **[Core Concepts](../2-core-concepts/index.md):** Understand the building blocks:
    -   [Transforms](../2-core-concepts/transforms.md): Individual augmentation operations.
    -   [Pipelines (Compose)](../2-core-concepts/pipelines.md): Sequencing multiple transforms.
    -   [Targets](../2-core-concepts/targets.md): Applying transforms to images, masks, bounding boxes, etc.
    -   [Setting Probabilities](../2-core-concepts/probabilities.md): Controlling the likelihood of applying transforms.
-   **[Basic Usage Examples](../3-basic-usage/index.md):** See how to apply augmentations for common tasks like:
    -   [Image Classification](../3-basic-usage/image-classification.md)
    -   [Semantic Segmentation](../3-basic-usage/semantic-segmentation.md)
    -   [Bounding Box Augmentation](../3-basic-usage/bounding-boxes-augmentations.md)
    -   [Keypoint Augmentation](../3-basic-usage/keypoint-augmentations.md)
-   **[Explore Transforms](https://explore.albumentations.ai):** Visually experiment with transforms and their parameters.

We hope you find Albumentations helpful for your projects!
