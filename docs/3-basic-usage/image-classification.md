# Image Classification with Albumentations

This guide demonstrates the practical steps for setting up and applying image augmentations for classification tasks using Albumentations. It focuses on the *how-to* aspects of defining and integrating augmentation pipelines.

For background on *why* data augmentation is crucial and *which* specific augmentations are effective for improving model generalization, please refer to these guides:

*   **[What is Data Augmentation?](../1-introduction/what-are-image-augmentations.md):** Explains the motivation and benefits.
*   **[Choosing Augmentations](./choosing-augmentations.md):** Provides detailed strategies for selecting and tuning various augmentations.

## Core Workflow

Applying augmentations typically involves these steps:

### 1. Setup: Import Libraries

Import Albumentations, an image reading library (like OpenCV), and any necessary framework components.

```python
import albumentations as A
import cv2
import numpy as np
```

### 2. Define Augmentation Pipelines

We use `A.Compose` to create a sequence of transformations. Separate pipelines are usually defined for training (with random augmentations) and validation/testing (with deterministic preprocessing).

**Example Training Pipeline:**

A common strategy involves resizing, cropping, basic geometric transforms, and normalization.

```python
TARGET_SIZE = 224 # Example input size

train_transform = A.Compose([
    # Resize shortest side to TARGET_SIZE, maintaining aspect ratio
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    # Take a random TARGET_SIZE x TARGET_SIZE crop
    A.RandomCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    # Apply horizontal flip with 50% probability
    A.HorizontalFlip(p=0.5),
    # Normalize using ImageNet presets
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Convert to PyTorch tensor format
    A.ToTensorV2(),
])
```

**Example Validation Pipeline:**

Typically includes resizing, center cropping, and normalization, without random elements.

```python
val_transform = A.Compose([
    A.SmallestMaxSize(max_size=TARGET_SIZE, p=1.0),
    # Take a crop from the center
    A.CenterCrop(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0),
    # Normalize using ImageNet presets
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Convert to PyTorch tensor format
    A.ToTensorV2(),
])
```

*Alternative:* [`A.RandomResizedCrop`](https://explore.albumentations.ai/transform/RandomResizedCrop) is another popular transform for training, combining resizing and cropping with scale/aspect ratio changes in one step.

### 3. Load Image Data

Load images into NumPy arrays. Remember that OpenCV reads images in BGR format by default, so convert to RGB if necessary.

```python
image_path = "/path/to/your/image.jpg"

# Read image and convert to RGB
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print(f"Loaded image shape: {image.shape}, dtype: {image.dtype}")
# Expected output e.g.: Loaded image shape: (512, 512, 3), dtype: uint8
```

### 4. Apply the Transform

The `Compose` object acts as a callable function. Pass the image as a keyword argument `image`. The output is a dictionary, with the transformed image under the `'image'` key.

```python
# Apply the training transform to a single image
augmented_data = train_transform(image=image)
augmented_image = augmented_data['image']

print(f"Augmented image shape: {augmented_image.shape}, dtype: {augmented_image.dtype}")
# Expected output e.g.: Augmented image shape: torch.Size([3, 224, 224]), dtype: torch.float32
```

### 5. Integrate into Framework Data Loader

In practice, you apply the transform within your deep learning framework's data loading pipeline (e.g., `torch.utils.data.Dataset` for PyTorch).

**Conceptual PyTorch `Dataset`:**

```python
from torch.utils.data import Dataset, DataLoader

class ClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform # Assign the A.Compose object here

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transforms
        if self.transform:
            # Pass image, get dictionary back
            augmented = self.transform(image=image)
            # Extract the transformed image tensor
            image = augmented['image']

        return image, label

# --- Usage Example ---
# Assuming train_paths, train_labels, val_paths, val_labels are defined
# train_dataset = ClassificationDataset(train_paths, train_labels, transform=train_transform)
# val_dataset = ClassificationDataset(val_paths, val_labels, transform=val_transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# # Training loop would iterate through train_loader
# for batch_images, batch_labels in train_loader:
#     # Model training steps...
#     pass
```

### 6. Visualize Augmentations (Crucial Debugging Step)

Always visualize the output of your *training* pipeline on sample images *before* starting a full training run. This helps verify that the transformations look reasonable and haven't corrupted the data.

**Important:** Visualize the output *before* applying `A.Normalize` and `A.ToTensorV2`, as these change the data type and value range, making direct display difficult.

```python
import matplotlib.pyplot as plt
import torch # For checking tensor type

def visualize_augmentations(dataset, idx=0, samples=5):
    # Make a copy of the transform list to modify for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        # Handle cases where transform might not be Compose (optional)
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(1, samples + 1, figsize=(12, 5))

    # --- Get the original image --- #
    # Temporarily disable transform to get raw image
    original_transform = dataset.transform
    dataset.transform = None
    image, label = dataset[idx]
    dataset.transform = original_transform # Restore original transform

    # Display original
    ax[0].imshow(image)
    ax[0].set_title("Original")
    ax[0].axis("off")

    # --- Apply and display augmented versions --- #
    for i in range(samples):
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image)
            aug_image = augmented['image']
        else:
             # Should not happen if dataset had a transform
            aug_image = image

        ax[i+1].imshow(aug_image)
        ax[i+1].set_title(f"Augmented {i+1}")
        ax[i+1].axis("off")

    plt.tight_layout()
    plt.show()

# Assuming train_dataset is created with train_transform:
# visualize_augmentations(train_dataset, samples=4)
```

# Apply the Test Pipeline
transformed_test = test_transform(image=image)
test_image_tensor = transformed_test["image"]

## Where to Go Next?

This guide covered the basic mechanics of applying augmentations for classification. To build more effective pipelines, explore the wide variety of transforms available in Albumentations and refer back to the **[Choosing Augmentations](./choosing-augmentations.md)** guide for detailed advice on selecting, combining, and tuning transforms to maximize your model's performance.

Here are some further resources to explore:

-   **[Learn How to Pick Augmentations](./choosing-augmentations.md):** Deepen your understanding of selecting effective transforms.
-   **[Optimize Performance](./performance-tuning.md):** Learn strategies to speed up your augmentation pipeline.
-   Explore Other Tasks: See how augmentations are handled with targets:
    -   [Semantic Segmentation](./semantic-segmentation.md)
    -   [Object Detection](./bounding-boxes-augmentations.md)
    -   [Keypoint Augmentation](./keypoint-augmentations.md)
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Browse the full range of available augmentations and their effects.
-   **[Review Core Concepts](../2-core-concepts):** Reinforce your understanding of the library's fundamentals.
-   **[Check Advanced Guides](../4-advanced-guides):** Look into topics like custom transforms or serialization.
