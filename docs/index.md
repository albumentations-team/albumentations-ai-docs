# Welcome to Albumentations Documentation!

Albumentations is a fast and flexible library for image augmentation. Whether you're working on classification, segmentation, object detection, or other computer vision tasks, Albumentations provides a comprehensive set of transforms and a powerful pipeline framework.

This documentation will guide you through installing the library, understanding its core concepts, applying it to various tasks, and exploring advanced features.

## Getting Started

*   **[Introduction](./1-introduction):** Learn what data augmentation is and why it's important.
*   **[Installation](./1-introduction/installation.md):** Set up Albumentations in your environment.

## Learning the Basics

*   **[Core Concepts](./2-core-concepts):** Understand the fundamental building blocks: Transforms, Pipelines (Compose), Targets (image, mask, bboxes, keypoints), and Probabilities.
*   **[Basic Usage Guides](./3-basic-usage):** Find practical examples for common computer vision tasks:
    *   [Image Classification](./3-basic-usage/image-classification.md)
    *   [Semantic Segmentation](./3-basic-usage/semantic-segmentation.md)
    *   [Object Detection (Bounding Boxes)](./3-basic-usage/bounding-boxes-augmentations.md)
    *   [Keypoint Augmentation](./3-basic-usage/keypoint-augmentations.md)
    *   [Video Augmentation](./3-basic-usage/video-augmentation.md)
    *   [Volumetric (3D) Augmentation](./3-basic-usage/volumetric-augmentation.md)
*   **[Choosing Augmentations](./3-basic-usage/choosing-augmentations.md):** A detailed guide on selecting effective augmentation strategies for model generalization.
*   **[Performance Tuning](./3-basic-usage/performance-tuning.md):** Tips for optimizing your augmentation pipeline speed.

## Advanced Topics

*   **[Advanced Guides](./4-advanced-guides):** Explore more complex features:
    *   [Using Additional Targets](./4-advanced-guides/additional-targets.md)
    *   [Creating Custom Transforms](./4-advanced-guides/creating-custom-transforms.md)
    *   [Serialization](./4-advanced-guides/serialization.md)

## Other Resources

*   **[Comparing with Torchvision/Kornia](./torchvision-kornia2albumentations.md):** See how Albumentations compares to other libraries.
*   **[Frequently Asked Questions (FAQ)](./faq.md):** Find answers to common questions.
*   **[Benchmarks](./benchmarks):** Performance comparison results.
*   **[Supported Targets by Transform](./reference/supported-targets-by-transform.md):** Check which transforms work with images, masks, bounding boxes, keypoints, etc.
*   **[API Reference](./api-reference)**
*   **[GitHub Repository](https://github.com/albumentations-team/albumentations):** Source code, issue tracking, and contributions.
*   **[Examples Folder (on GitHub)](https://github.com/albumentations-team/albumentations_examples):** Many practical examples in the main repository.

We hope this documentation helps you leverage the full power of Albumentations!
