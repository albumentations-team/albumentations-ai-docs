# Quick Start Guide

Get up and running with Albumentations in 5 minutes! This guide shows you how to install the library and create your first augmentation pipeline.

## Install Albumentations

```bash
pip install albumentations
```

## Your First Pipeline (30 seconds)

```python
import albumentations as A
import cv2
import numpy as np

# Create augmentation pipeline
transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

# Load an image
image = cv2.imread("path/to/your/image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply augmentations
augmented = transform(image=image)
augmented_image = augmented['image']
```

That's it! You now have an augmented image.

## Common Patterns for Different Tasks

### Image Classification
```python
import albumentations as A

# Training pipeline
train_transform = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

# Validation pipeline
val_transform = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
```

### Object Detection (with bboxes)
```python
import albumentations as A

transform = A.Compose([
    A.RandomSizedBBoxSafeCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Apply to image and bboxes
augmented = transform(
    image=image,
    bboxes=bboxes,
    class_labels=class_labels
)
```

### Semantic Segmentation (with masks)
```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),  # Only affects image
])

# Apply to image and mask
augmented = transform(image=image, mask=mask)
augmented_image = augmented['image']
augmented_mask = augmented['mask']
```

## PyTorch Integration

```python
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.labels[idx]

# Create dataset with augmentations
dataset = ImageDataset(image_paths, labels, transform=train_transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Key Concepts (2 minutes)

### Probability Control
Every transform has a `p` parameter controlling application probability:
```python
A.HorizontalFlip(p=0.5)    # 50% chance to flip
A.HorizontalFlip(p=1.0)    # Always flip
A.HorizontalFlip(p=0.0)    # Never flip
```

### Transform Categories
- **Spatial transforms**: Change geometry (flip, rotate, crop) → affect all targets
- **Pixel transforms**: Change colors/intensity → only affect images

### Multiple Targets
Pass multiple targets to keep them synchronized:
```python
result = transform(image=image, mask=mask, bboxes=bboxes, keypoints=keypoints)
```

## Useful Tools

### Visualize Augmentations
```python
import matplotlib.pyplot as plt

# Quick visualization function
def show_augmentations(image, transform, count=4):
    fig, axes = plt.subplots(1, count, figsize=(15, 5))
    for i in range(count):
        augmented = transform(image=image)['image']
        axes[i].imshow(augmented)
        axes[i].axis('off')
    plt.show()

show_augmentations(image, transform)
```

### Interactive Exploration
Visit [explore.albumentations.ai](https://explore.albumentations.ai) to:
- Upload your images
- Test transforms visually
- Experiment with parameters
- Copy working code

## Essential Performance Tips

1. **Crop early**: Put cropping transforms first in your pipeline
```python
# ✅ Good - fast
A.Compose([
    A.RandomCrop(224, 224),  # First!
    A.HorizontalFlip(),
    A.GaussianBlur(),
])
```

2. **Use multiple DataLoader workers**:
```python
DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
```

3. **Check transform compatibility**: Use the [Supported Targets by Transform](../reference/supported-targets-by-transform.md) reference to verify which transforms work with your data types.

## Ready for More?

You now have a working augmentation pipeline! Here's where to go next:

### Immediate Next Steps:
-   **[Choose Better Augmentations](../3-basic-usage/choosing-augmentations.md):** Learn systematic approaches to selecting transforms for maximum performance
-   **[Your Specific Task](../3-basic-usage/index.md):** Deep dive into classification, detection, segmentation, etc.

### Core Understanding:
-   **[Core Concepts](../2-core-concepts/index.md):** Understand transforms, pipelines, targets, and probabilities
-   **[Performance Optimization](../3-basic-usage/performance-tuning.md):** Make your pipelines faster

### Exploration:
-   **[All Available Transforms](https://explore.albumentations.ai):** Interactive exploration tool
-   **[Advanced Features](../4-advanced-guides/index.md):** Custom transforms, serialization, additional targets

**Pro tip**: Start simple with basic transforms, then gradually add complexity based on your validation results!
