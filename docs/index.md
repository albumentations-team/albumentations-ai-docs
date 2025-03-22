# Welcome to Albumentations documentation

Albumentations is a fast and flexible image augmentation library. The library is widely used in [industry](https://albumentations.ai/whos_using#industry), [deep learning research](https://albumentations.ai/whos_using#research), [machine learning competitions](https://albumentations.ai/whos_using#competitions), and [open source projects](https://albumentations.ai/whos_using#open-source). Albumentations is written in Python, and it is licensed under the MIT license. The source code is available at [https://github.com/albumentations-team/albumentations](https://github.com/albumentations-team/albumentations).

## Getting Started

- [Installation](getting_started/installation.md)
- [Frequently Asked Questions](faq.md)
- [Your First Augmentation Pipeline](examples/example/)

You can also visit [explore.albumentations.ai](https://explore.albumentations.ai) to visually explore and experiment with different augmentations in your browser. This interactive tool helps you better understand how each transform affects images before implementing it in your code.

## Learning Path

### Beginners

- [What is Image Augmentation?](introduction/image_augmentation.md)
- [Why Choose Albumentations?](introduction/why_albumentations.md)
- [Basic Image Classification](getting_started/image_augmentation.md)

### Intermediate

- [Semantic Segmentation](getting_started/mask_augmentation.md)
- [Object Detection](getting_started/bounding_boxes_augmentation.md)
- [Keypoint Detection](getting_started/keypoints_augmentation.md)
- [Multi-target Augmentation](getting_started/simultaneous_augmentation.md)
- [Transforms and Targets](api_reference/full_reference.md)

### Advanced

- [Pipeline Configuration](getting_started/setting_probabilities.md)
- [Debugging with ReplayCompose](examples/replay/)
- [Serialization](examples/serialization/)

## Working with Multi-dimensional Data

### Volumetric Data (3D)
- [Introduction to 3D (Volumetric) Image Augmentation](getting_started/volumetric_augmentation.md)
- [Available 3D Transforms](api_reference/augmentations/transforms3d/index.md)

### Video and Sequential Data
- [Video Frame Augmentation](getting_started/video_augmentation.md)

## Framework Integration

- [PyTorch](examples/pytorch_classification/)
- [HuggingFace](integrations/huggingface/)
- [Voxel51](integrations/fiftyone.md)

## Library Comparisons

- [Transform Library Comparison](getting_started/augmentation_mapping.md) - Find equivalent transforms between Albumentations and other libraries (torchvision, Kornia)
- [Migration from torchvision](examples/migrating_from_torchvision_to_albumentations/) - Step-by-step migration guide

## Examples by Task

- [Image Classification](examples/example/)
- [Object Detection](examples/example_bboxes/)
- [Object Detection (Preserving All Boxes)](examples/example_bboxes2/)
- [Semantic Segmentation](examples/example_kaggle_salt/)
- [Keypoint Detection](examples/example_keypoints/)
- [Multi-target Augmentation](examples/example_multi_target/)

## Transform Examples

- [Weather Augmentations](examples/example_weather_transforms/)
- [XYMasking Transform](examples/example_xymasking/)
- [Chromatic Aberration Transform](examples/example_chromatic_aberration/)
- [Morphological Transform](examples/example_documents/)
- [D4 Transform](examples/example_d4/)
- [RandomGridShuffle Transform](examples/example_gridshuffle/)
- [OverlayElements Transform](examples/example_OverlayElements/)
- [TextImage Transform](examples/example_textimage/)

## Advanced Usage

- [Debugging with ReplayCompose](examples/replay/)
- [Serialization](examples/serialization/)
- [Showcase of Diverse Augmentations](examples/showcase/)
- [HuggingFace Hub Integration](examples/example_hfhub/)
- [PyTorch Integration Examples](examples/pytorch_classification/)
- [PyTorch Semantic Segmentation](examples/pytorch_semantic_segmentation/)

## External Resources

- [Blog Posts, Podcasts, Talks, and Videos](external_resources/blog_posts_podcasts_talks.md)
- [Books](external_resources/books.md)
- [Online Courses](external_resources/online_courses.md)

## Other Topics

- [Contributing](CONTRIBUTING.md)

## API Reference

- [Full API Reference on a Single Page](api_reference/full_reference.md)
- [Index](api_reference/index.md)
  - [Core API (albumentations.core)](api_reference/core/index.md)
  - [Augmentations (albumentations.augmentations)](api_reference/augmentations/index.md)
  - [PyTorch Helpers (albumentations.pytorch)](api_reference/pytorch/index.md)
