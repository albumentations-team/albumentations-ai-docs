# Welcome to Albumentations documentation

Albumentations is a fast and flexible image augmentation library. The library is widely used in [industry](https://albumentations.ai/whos_using#industry), [deep learning research](https://albumentations.ai/whos_using#research), [machine learning competitions](https://albumentations.ai/whos_using#competitions), and [open source projects](https://albumentations.ai/whos_using#open-source). Albumentations is written in Python, and it is licensed under the MIT license. The source code is available at [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations).

If you are new to image augmentation, start with our ["Learning Path"](#learning-path) for beginners. It describes what image augmentation is, how it can boost deep neural networks' performance, and why you should use Albumentations.

For hands-on experience, check out our ["Quick Start Guide"](#quick-start-guide) and ["Examples"](#examples) sections. They show how you can use the library for different computer vision tasks: image classification, semantic segmentation, instance segmentation, object detection, and keypoint detection. Each example includes a link to Google Colab, where you can run the code by yourself.

You can also visit [explore.albumentations.ai](https://explore.albumentations.ai) to visually explore and experiment with different augmentations in your browser. This interactive tool helps you better understand how each transform affects images before implementing it in your code.

["API Reference"](#api-reference) contains the description of Albumentations' methods and classes.

## Quick Start Guide

- [Installation](getting-started/installation.md)
- [Frequently Asked Questions](faq.md)
- [Your First Augmentation Pipeline](examples/example/)

## Working with Multi-dimensional Data

### Volumetric Data (3D)
- [Introduction to 3D (Volumetric) Image Augmentation](getting-started/volumetric-augmentation.md)
- [Available 3D Transforms](api-reference/augmentations/transforms3d/index.md)

### Video and Sequential Data
- [Video Frame Augmentation](getting-started/video-augmentation.md)

## Learning Path

### Beginners

- [What is Image Augmentation?](introduction/image-augmentation.md)
- [Why Choose Albumentations?](introduction/why-albumentations.md)
- [Basic Image Classification](getting-started/image-augmentation.md)

### Intermediate

- [Semantic Segmentation](getting-started/mask-augmentation.md)
- [Object Detection](getting-started/bounding-boxes-augmentation.md)
- [Keypoint Detection](getting-started/keypoints-augmentation.md)
- [Multi-target Augmentation](getting-started/simultaneous-augmentation.md)

### Advanced

- [Pipeline Configuration](getting-started/setting-probabilities.md)
- [Debugging with ReplayCompose](examples/replay/)
- [Serialization](examples/serialization/)

## Library Comparisons

- [Transform Library Comparison](getting-started/augmentation-mapping.md) - Find equivalent transforms between Albumentations and other libraries (torchvision, Kornia)
- [Migration from torchvision](examples/migrating-from-torchvision-to-albumentations/) - Step-by-step migration guide

## Examples

- [Defining a simple augmentation pipeline for image augmentation](examples/example/)
- [Using Albumentations to augment bounding boxes for object detection tasks](examples/example-bboxes/)
- [How to use Albumentations for detection tasks if you need to keep all bounding boxes](examples/example-bboxes2/)
- [Using Albumentations for a semantic segmentation task](examples/example-kaggle-salt/)
- [Using Albumentations to augment keypoints](examples/example-keypoints/)
- [Applying the same augmentation with the same parameters to multiple images, masks, bounding boxes, or keypoints](examples/example-multi-target/)
- [Weather augmentations in Albumentations](examples/example-weather-transforms/)
- [Example of applying XYMasking transform](examples/example-xymasking/)
- [Example of applying ChromaticAberration transform](examples/example-chromatic-aberration/)
- [Example of applying Morphological transform](examples/example-documents/)
- [Example of applying D4 transform](examples/example-d4/)
- [Example of applying RandomGridShuffle transform](examples/example-gridshuffle/)
- [Example of applying OverlayElements transform](examples/example-OverlayElements/)
- [Example of applying TextImage transform](examples/example-textimage/)
- [Debugging an augmentation pipeline with ReplayCompose](examples/replay/)
- [How to save and load parameters of an augmentation pipeline](examples/serialization/)
- [Showcase. Cool augmentation examples on diverse set of images from various real-world tasks.](examples/showcase/)
- [How to save and load transforms to HuggingFace Hub.](examples/example-hfhub/)

## Examples of how to use Albumentations with different deep learning frameworks

- [PyTorch and Albumentations for image classification](examples/pytorch-classification/)
- [PyTorch and Albumentations for semantic segmentation](examples/pytorch-semantic-segmentation/)

## External resources

- [Blog posts, podcasts, talks, and videos about Albumentations](external-resources/blog-posts-podcasts-talks.md)
- [Books that mention Albumentations](external-resources/books.md)
- [Online courses that cover Albumentations](external-resources/online-courses.md)

## Other topics

- [Contributing](CONTRIBUTING.md)

## API Reference

- [Full API Reference on a single page](api-reference/full-reference.md)
- [Index](api-reference/index.md)
  - [Core API (albumentations.core)](api-reference/core/index.md)
  - [Augmentations (albumentations.augmentations)](api-reference/augmentations/index.md)
  - [PyTorch Helpers (albumentations.pytorch)](api-reference/pytorch/index.md)
