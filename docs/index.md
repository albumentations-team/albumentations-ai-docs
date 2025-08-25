# Welcome to Albumentations Documentation!

> 🚀 **Important: AlbumentationsX - The Next Generation**
>
> **[AlbumentationsX](https://github.com/albumentations-team/AlbumentationsX)** is now available as the actively maintained successor to Albumentations:
>
> - ✅ **100% drop-in replacement** - no code changes required
> - ⚡ **Better performance** with bug fixes and new features
> - 🔧 **Active maintenance** and professional support
> - 📄 **Licensed** under AGPL/Commercial - see our [License Guide](./license.md)
>
> ```bash
> pip uninstall albumentations  # If you have it installed
> pip install albumentationsx   # Install the new version
> ```
>
> Your existing code continues to work exactly as before! Learn more in our [Installation Guide](./1-introduction/installation.md).

Albumentations is a fast and flexible library for image augmentation. Whether you're working on classification, segmentation, object detection, or other computer vision tasks, Albumentations provides a comprehensive set of transforms and a powerful pipeline framework.

This documentation will guide you through installing the library, understanding its core concepts, applying it to various tasks, and exploring advanced features.

## Getting Started

*   **[Introduction](./1-introduction/index.md):** Learn what data augmentation is and why it's important.
*   **[Installation](./1-introduction/installation.md):** Set up Albumentations in your environment.

## Learning the Basics

*   **[Core Concepts](./2-core-concepts/index.md):** Understand the fundamental building blocks: Transforms, Pipelines (Compose), Targets (image, mask, bboxes, keypoints), and Probabilities.
*   **[Basic Usage Guides](./3-basic-usage/index.md):** Find practical examples for common computer vision tasks:
    *   [Image Classification](./3-basic-usage/image-classification.md)
    *   [Semantic Segmentation](./3-basic-usage/semantic-segmentation.md)
    *   [Object Detection (Bounding Boxes)](./3-basic-usage/bounding-boxes-augmentations.md)
    *   [Keypoint Augmentation](./3-basic-usage/keypoint-augmentations.md)
    *   [Video Augmentation](./3-basic-usage/video-augmentation.md)
    *   [Volumetric (3D) Augmentation](./3-basic-usage/volumetric-augmentation.md)
*   **[Choosing Augmentations](./3-basic-usage/choosing-augmentations.md):** A detailed guide on selecting effective augmentation strategies for model generalization.
*   **[Performance Tuning](./3-basic-usage/performance-tuning.md):** Tips for optimizing your augmentation pipeline speed.

## Advanced Topics

*   **[Advanced Guides](./4-advanced-guides/index.md):** Explore more complex features:
    *   [Using Additional Targets](./4-advanced-guides/additional-targets.md)
    *   [Creating Custom Transforms](./4-advanced-guides/creating-custom-transforms.md)
    *   [Serialization](./4-advanced-guides/serialization.md)

## Other Resources

*   **[Comparing with Torchvision/Kornia](./torchvision-kornia2albumentations.md):** See how Albumentations compares to other libraries.
*   **[Frequently Asked Questions (FAQ)](./faq.md):** Find answers to common questions.
*   **[Benchmarks](./benchmarks):** Performance comparison results.
*   **[Supported Targets by Transform](./reference/supported-targets-by-transform.md):** Check which transforms work with images, masks, bounding boxes, keypoints, etc.
*   **[API Reference](./api-reference)**
*   **[GitHub Repository](https://github.com/albumentations-team/AlbumentationsX):** Active development
*   **[Examples Repository](https://github.com/albumentations-team/albumentations_examples):** Many practical examples.

We hope this documentation helps you leverage the full power of Albumentations!
