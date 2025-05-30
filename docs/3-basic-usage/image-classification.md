# Image Classification with Albumentations

This guide shows how to set up practical augmentation pipelines for image classification training. We'll cover the essential patterns: MNIST-style (grayscale) and ImageNet-style (RGB) pipelines with actual training integration.

For background on *why* data augmentation is crucial and *which* specific augmentations are effective, please refer to:

*   **[What is Data Augmentation?](../1-introduction/what-are-image-augmentations.md):** Explains the motivation and benefits.
*   **[Choosing Augmentations](./choosing-augmentations.md):** Detailed strategies for selecting and tuning transforms.

## Quick Reference

**Classification Pipeline Essentials:**
- **Image-only transforms**: No need to worry about masks, bboxes, or keypoints
- **Training pipeline**: Random augmentations for variety
- **Validation pipeline**: Deterministic preprocessing only
- **Common pattern**: Resize → Crop → Augment → Normalize → Tensor

**Minimal Training Pipeline:**
```python
import albumentations as A

train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
```

## Core Workflow

### 1. Import Libraries

```python
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
```

### 2. Define Transform Pipelines

#### MNIST-Style Pipeline (Grayscale, 28x28)

Simple pipeline for grayscale images like MNIST, Fashion-MNIST, or custom grayscale datasets:

```python
# Training transforms - minimal augmentation for small images
train_transforms = A.Compose([
    A.Resize(32, 32),  # Slightly larger than target
    A.RandomCrop(28, 28),
    A.Rotate(limit=10, p=0.5),  # Small rotation
    A.Normalize(mean=[0.1307], std=[0.3081]),  # MNIST stats
    A.ToTensorV2(),
])

# Validation transforms - deterministic
val_transforms = A.Compose([
    A.Resize(28, 28),
    A.Normalize(mean=[0.1307], std=[0.3081]),
    A.ToTensorV2(),
])
```

#### ImageNet-Style Pipeline (RGB, 224x224)

Standard pipeline for RGB images, following ImageNet preprocessing:

```python
# Training transforms - comprehensive augmentation
train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])

# Validation transforms - deterministic
val_transforms = A.Compose([
    A.Resize(256, 256),
    A.CenterCrop(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
```

### 3. Create Dataset Class

Simple PyTorch dataset that integrates Albumentations:

```python
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, self.labels[idx]
```

### 4. Training Integration Examples

#### MNIST-Style Training

```python
# Create datasets
train_dataset = ImageClassificationDataset(train_paths, train_labels, train_transforms)
val_dataset = ImageClassificationDataset(val_paths, val_labels, val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Simple training loop
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Training
model.train()
for epoch in range(10):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

#### ImageNet-Style Training

```python
import torchvision.models as models

# Create datasets with ImageNet-style transforms
train_dataset = ImageClassificationDataset(train_paths, train_labels, train_transforms)
val_dataset = ImageClassificationDataset(val_paths, val_labels, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Use pre-trained model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()

    print(f'Epoch {epoch}: Val Loss: {val_loss/len(val_loader):.4f}, '
          f'Val Acc: {correct/len(val_dataset):.4f}')
```

### 5. Quick Validation Check

Always verify your transforms work correctly before training:

```python
# Quick check - load one image and see the output
sample_image = cv2.imread(train_paths[0])
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

augmented = train_transforms(image=sample_image)
print(f"Input shape: {sample_image.shape}")
print(f"Output shape: {augmented['image'].shape}")
print(f"Output type: {type(augmented['image'])}")
# Expected: torch.Size([3, 224, 224]) for ImageNet-style
```

## Advanced Pipeline Examples

### Stronger Augmentation Pipeline

For when you need more regularization:

```python
strong_train_transforms = A.Compose([
    A.RandomResizedCrop(224, 224, scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        A.ToGray(p=1.0),
    ], p=0.8),
    A.OneOf([
        A.GaussianBlur(blur_limit=(1, 3)),
        A.GaussNoise(var_limit=(10, 50)),
    ], p=0.5),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.ToTensorV2(),
])
```

### Domain-Specific Pipeline

For medical images or other specialized domains:

```python
medical_transforms = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.SquareSymmetry(p=0.5),  # All 8 rotations/flips - proper for medical data
    A.ElasticTransform(alpha=50, sigma=5, p=0.3),  # Tissue-like distortion
    A.Normalize(mean=[0.5], std=[0.5]),  # Center around 0
    A.ToTensorV2(),
])
```

## Performance Tips

1. **Use `num_workers`** in DataLoader for faster loading
2. **Pin memory** with `pin_memory=True` if using GPU
3. **Cache transforms** - don't recreate Compose objects in loops
4. **Profile your pipeline** - augmentation shouldn't be the bottleneck

```python
# Optimized data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel loading
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

## Transform Selection Guide

Classification is the simplest case since we only deal with images (no masks, bboxes, or keypoints). All transforms support image-only operation. For a comprehensive list of available transforms and their effects, see the [Supported Targets by Transform](../reference/supported-targets-by-transform.md) reference.

**Essential transforms for classification:**
- **Scale/crop**: [`RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop), [`Resize`](https://explore.albumentations.ai/transform/Resize)
- **Flip**: [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip) (be careful with [`VerticalFlip`](https://explore.albumentations.ai/transform/VerticalFlip) - not suitable for natural images)
- **Color**: [`ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter), [`RandomBrightnessContrast`](https://explore.albumentations.ai/transform/RandomBrightnessContrast)
- **Regularization**: [`CoarseDropout`](https://explore.albumentations.ai/transform/CoarseDropout), [`ToGray`](https://explore.albumentations.ai/transform/ToGray)

## Where to Go Next?

Now that you have working classification pipelines:

-   **[Optimize Your Augmentation Strategy](./choosing-augmentations.md):** Learn systematic approaches to selecting and tuning transforms for maximum performance.
-   **[Performance Tuning](./performance-tuning.md):** Speed up your training pipeline and reduce data loading bottlenecks.
-   **[Explore More Complex Tasks](../3-basic-usage/):** See how augmentations work with multiple targets:
    -   [Object Detection](./bounding-boxes-augmentations.md) - handling bounding boxes
    -   [Semantic Segmentation](./semantic-segmentation.md) - working with masks
    -   [Keypoint Detection](./keypoint-augmentations.md) - preserving point annotations
-   **[Advanced Techniques](../4-advanced-guides/):** Custom transforms, serialization, and specialized augmentation strategies.
-   **[Interactive Exploration](https://explore.albumentations.ai):** Visually experiment with transforms on your own images.
-   **[Core Concepts](../2-core-concepts/):** Deepen your understanding of [transforms](../2-core-concepts/transforms.md) and [pipelines](../2-core-concepts/pipelines.md).
