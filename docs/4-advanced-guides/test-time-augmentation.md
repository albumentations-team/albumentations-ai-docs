# Test Time Augmentation (TTA)

Test Time Augmentation (TTA) is a technique where you apply augmentations at **inference time** rather than (or in addition to) training time, and **average the predictions** over multiple augmented versions of the same input. It is a simple, model-agnostic way to improve prediction quality without retraining.

## Train-Time vs Test-Time Augmentation

During **training**, we apply random augmentations to each image before feeding it into the network. The goal is to expose the model to a wider variety of inputs so that it learns features that are robust to those variations. Each training step sees one randomly augmented version of each sample.

During **inference** (test time), we typically apply only deterministic preprocessing — resize, center crop, normalize. We get a single prediction per image.

**Test Time Augmentation** bridges the gap: at inference, we create **multiple augmented versions** of the same input, pass each through the network, and **aggregate** the predictions. The idea is straightforward — if the network has seen similar variations during training, it should produce consistent (but slightly different) predictions for each variation. Averaging these predictions reduces variance and often leads to a small but meaningful accuracy improvement.

```
                    ┌─── Transform 1 ──→ Network ──→ Prediction 1 ───┐
                    │                                                  │
Input Image ────────┼─── Transform 2 ──→ Network ──→ Prediction 2 ───┼──→ Average ──→ Final Prediction
                    │                                                  │
                    └─── Transform 3 ──→ Network ──→ Prediction 3 ───┘
```

## TTA for Image Classification

### Invariance: What We Expect from the Network

For classification, we want the network to be **invariant** to certain transformations — the class label should not change when we flip, slightly rotate, or adjust the color of an image. A photo of a cat is still a cat when flipped horizontally.

Formally, for a classification network $f$ and a transformation $T$:

$$f(T(x)) = f(x)$$

In practice, neural networks are only *approximately* invariant. TTA exploits this by averaging predictions across transformed versions, pushing the final result closer to the true invariant output.

### Which Transformations for ImageNet?

For natural images (like ImageNet), the set of reasonable TTA transforms reflects the natural symmetries of the data:

- **[HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)** — The most common and cheapest TTA. Natural scenes are roughly symmetric under left-right reflection. This is a symmetry group with **2 elements** (identity + flip).
- **Small [Affine](https://explore.albumentations.ai/transform/Affine) transforms** — Slight translations, rotations (±5°), and scale changes. Objects don't change class with minor geometric shifts.
- **Small color variations** — [`ColorJitter`](https://explore.albumentations.ai/transform/ColorJitter), [`RandomGamma`](https://explore.albumentations.ai/transform/RandomGamma), [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise). Lighting conditions vary in the real world; the network should be robust to these.

### Averaging: Logits vs Probabilities

When aggregating predictions from multiple augmented views, you have two choices:

**Option 1: Average logits (pre-softmax)**
```python
logits_list = [model(transform(image)) for transform in tta_transforms]
avg_logits = torch.stack(logits_list).mean(dim=0)
prediction = avg_logits.argmax()
```

**Option 2: Average probabilities (post-softmax)**
```python
probs_list = [F.softmax(model(transform(image)), dim=-1) for transform in tta_transforms]
avg_probs = torch.stack(probs_list).mean(dim=0)
prediction = avg_probs.argmax()
```

Both approaches work. In practice, **averaging logits** is slightly more common because:

- It is equivalent to a geometric mean in probability space (after softmax), which gives less weight to overconfident wrong predictions.
- It avoids applying softmax multiple times.

The difference is usually negligible, but averaging logits is the more principled default.

### Standard TTA Strategies for Classification

#### [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip) TTA (2 views)

The simplest and most universal TTA. Average predictions from the original and horizontally flipped image:

```python
import albumentations as A
import torch

def horizontal_flip_tta(model, image, preprocess):
    """TTA with original + horizontally flipped image."""
    # Original
    orig = preprocess(image=image)["image"]

    # Flipped
    flipped = A.HorizontalFlip(p=1.0)(image=image)["image"]
    flip_processed = preprocess(image=flipped)["image"]

    # Average logits
    with torch.no_grad():
        logits_orig = model(orig.unsqueeze(0))
        logits_flip = model(flip_processed.unsqueeze(0))
        avg_logits = (logits_orig + logits_flip) / 2

    return avg_logits
```

#### FiveCrop TTA (5 views)

Extract 5 crops from the image (4 corners + center) and average. Since Albumentations processes one image per call, we define the 5 cropping regions and apply them individually:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

def five_crop_tta(model, image):
    """TTA with 5 crops: 4 corners + center using Albumentations."""
    # First resize the image (e.g., SmallestMaxSize to 256)
    image_resized = A.SmallestMaxSize(max_size=256)(image=image)["image"]
    h, w = image_resized.shape[:2]
    crop_size = 224

    # Define the 5 crop boxes: [x_min, y_min, x_max, y_max]
    boxes = [
        [0, 0, crop_size, crop_size],                                  # Top-Left
        [w - crop_size, 0, w, crop_size],                              # Top-Right
        [0, h - crop_size, crop_size, h],                              # Bottom-Left
        [w - crop_size, h - crop_size, w, h],                          # Bottom-Right
        [(w - crop_size)//2, (h - crop_size)//2,
         (w + crop_size)//2, (h + crop_size)//2]                       # Center
    ]

    normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    crops = []
    for x_min, y_min, x_max, y_max in boxes:
        # Extract crop
        crop = A.Crop(x_min=x_min, y_min=y_min,
                      x_max=x_max, y_max=y_max)(image=image_resized)["image"]
        # Normalize and convert to tensor
        crop_tensor = normalize(image=crop)["image"]
        crops.append(crop_tensor)

    crop_tensors = torch.stack(crops)

    with torch.no_grad():
        logits = model(crop_tensors)  # [5, num_classes]
        avg_logits = logits.mean(dim=0, keepdim=True)

    return avg_logits
```

#### TenCrop TTA (10 views)

FiveCrop + their horizontal flips:

```python
def ten_crop_tta(model, image):
    """TTA with 10 crops: FiveCrop + their horizontal flips."""
    image_resized = A.SmallestMaxSize(max_size=256)(image=image)["image"]
    h, w = image_resized.shape[:2]
    crop_size = 224

    boxes = [
        [0, 0, crop_size, crop_size],
        [w - crop_size, 0, w, crop_size],
        [0, h - crop_size, crop_size, h],
        [w - crop_size, h - crop_size, w, h],
        [(w - crop_size)//2, (h - crop_size)//2, (w + crop_size)//2, (h + crop_size)//2]
    ]

    normalize = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    hflip = A.HorizontalFlip(p=1.0)

    crops = []
    for x_min, y_min, x_max, y_max in boxes:
        # Extract crop
        crop = A.Crop(x_min=x_min, y_min=y_min,
                      x_max=x_max, y_max=y_max)(image=image_resized)["image"]

        # Original crop
        crops.append(normalize(image=crop)["image"])

        # Flipped crop
        flipped_crop = hflip(image=crop)["image"]
        crops.append(normalize(image=flipped_crop)["image"])

    crop_tensors = torch.stack(crops)

    with torch.no_grad():
        logits = model(crop_tensors)  # [10, num_classes]
        avg_logits = logits.mean(dim=0, keepdim=True)

    return avg_logits
```

### Benchmark: ResNet18 on ImageNet Validation

We evaluated TTA strategies using a pretrained ResNet18 from [timm](https://github.com/huggingface/pytorch-image-models) on the full ImageNet validation set (50,000 images):

| Method | Top-1 Accuracy | Top-5 Accuracy | Views | Relative Slowdown |
|---|---|---|---|---|
| Baseline ([CenterCrop](https://explore.albumentations.ai/transform/CenterCrop)) | 71.19% | 89.79% | 1× | 1.0× |
| [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip) TTA | 71.91% | 90.25% | 2× | ~2.8× |
| FiveCrop TTA | 72.70% | 90.93% | 5× | ~6.3× |
| TenCrop TTA | 73.12% | 91.15% | 10× | ~8.8× |

> [!NOTE]
> These results were generated by evaluating the full 50,000 image ImageNet validation set. As expected, [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip) achieves an easy +0.72% gain on baseline. TenCrop pushes raw impact almost +2.0% top-1, demonstrating the classic diminishing returns curve against inference cost.

The full benchmark script is available at [`scripts/tta_imagenet_benchmark.py`](../../scripts/tta-imagenet-benchmark.py).

```bash
# Run the full benchmark
python scripts/tta_imagenet_benchmark.py --data-dir ~/data/imagenet/val

# Quick test on 1000 images
python scripts/tta_imagenet_benchmark.py --num-images 1000
```

## TTA for Semantic Segmentation

### Equivariance: What We Expect from the Network

Semantic segmentation assigns a class label **to every pixel**. You can think of it as running a classifier independently for each pixel location. So the output is not a single vector, but a **spatial map** of class predictions — one classifier per pixel.

This means TTA for segmentation works similarly to classification, but with an important twist. Instead of invariance, we need **equivariance**: if we transform the input image, the output segmentation map should transform in the same way.

Formally, for a segmentation network $f$ and a geometric transformation $T$ with inverse $T^{-1}$:

$$T^{-1}(f(T(x))) = f(x)$$

If we horizontally flip the input image and run it through the network, the resulting segmentation map should be the horizontally flipped version of what we would get without flipping.

### The TTA Schema for Segmentation

The averaging process for segmentation is:

1. Take the original image, run it through the network → get prediction map
2. Apply a geometric transform $T$ to the image
3. Run the transformed image through the network → get transformed prediction map
4. Apply the **inverse transform** $T^{-1}$ to the prediction map to align it back
5. Average the aligned prediction maps (per-pixel logits or probabilities)

![TTA equivariance schema for semantic segmentation](../../img/tta_equivariance_segmentation.webp)

```python
import albumentations as A
import numpy as np
import torch

def segmentation_hflip_tta(model, image, preprocess):
    """TTA for segmentation with HorizontalFlip.

    Key insight: we must apply the INVERSE transform to the prediction
    before averaging, to align predictions in the same coordinate space.
    """
    # Original prediction
    input_orig = preprocess(image=image)["image"].unsqueeze(0)
    with torch.no_grad():
        pred_orig = model(input_orig)  # [1, C, H, W]

    # Flipped prediction
    flipped = A.HorizontalFlip(p=1.0)(image=image)["image"]
    input_flip = preprocess(image=flipped)["image"].unsqueeze(0)
    with torch.no_grad():
        pred_flip = model(input_flip)  # [1, C, H, W]

    # Apply INVERSE transform (flip back) to the flipped prediction
    pred_flip_aligned = torch.flip(pred_flip, dims=[-1])  # flip width axis

    # Average aligned predictions
    avg_prediction = (pred_orig + pred_flip_aligned) / 2

    return avg_prediction
```

> [!IMPORTANT]
> The critical step that distinguishes segmentation TTA from classification TTA is applying the **inverse geometric transform** to the prediction map before averaging. For [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip), the inverse is another [`HorizontalFlip`](https://explore.albumentations.ai/transform/HorizontalFlip). For rotation by angle $\theta$, the inverse is rotation by $-\theta$. Color transforms (brightness, contrast) don't have spatial inverses — they only affect the input, not the output map.


## TTA for Object Detection

For object detection — including one-shot detectors like YOLO — the story is the same as segmentation. If the problem has a geometric symmetry, we expect the network to be **equivariant**: applying a transform to the input should produce correspondingly transformed bounding boxes.

The TTA workflow for detection:

1. Apply transform $T$ to the input image
2. Run the detector → get bounding boxes in the transformed coordinate space
3. Apply inverse transform $T^{-1}$ to the predicted bounding boxes
4. Merge predictions from all views (using NMS or weighted box fusion)

```python
def detection_hflip_tta(model, image):
    """TTA for object detection with HorizontalFlip."""
    width = image.shape[1]

    # Original predictions
    boxes_orig, scores_orig, labels_orig = model(image)

    # Flipped predictions
    flipped = image[:, ::-1, :]  # horizontal flip
    boxes_flip, scores_flip, labels_flip = model(flipped)

    # Inverse transform: flip bounding boxes back
    # For HorizontalFlip: x_new = width - x_old (swap x_min and x_max)
    boxes_flip_aligned = boxes_flip.copy()
    boxes_flip_aligned[:, 0] = width - boxes_flip[:, 2]  # new x_min = width - old x_max
    boxes_flip_aligned[:, 2] = width - boxes_flip[:, 0]  # new x_max = width - old x_min

    # Merge all predictions and run NMS
    all_boxes = np.concatenate([boxes_orig, boxes_flip_aligned])
    all_scores = np.concatenate([scores_orig, scores_flip])
    all_labels = np.concatenate([labels_orig, labels_flip])

    # Apply NMS or Weighted Box Fusion to merge overlapping detections
    final_boxes, final_scores, final_labels = weighted_box_fusion(
        all_boxes, all_scores, all_labels
    )
    return final_boxes, final_scores, final_labels
```

## Symmetry Groups Across Domains

Different tasks and image domains have different natural symmetries, which determines which TTA transforms are appropriate:

### Natural Images (ImageNet, COCO, etc.)

- **[HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)** — the primary geometric symmetry. Natural scenes are roughly left-right symmetric(a person facing left vs right is still a person). This forms a **symmetry group with 2 elements** (identity + flip).
- **[VerticalFlip](https://explore.albumentations.ai/transform/VerticalFlip)** — ⚠️ generally **not appropriate** for natural images. The world has a strong gravitational prior — sky is up, ground is down.
- **Small affine transforms** — slight translations, rotations (±5°), scale changes.
- **Color variations** — brightness, contrast, gamma, noise.

### Satellite / Aerial Imagery (Top-Down View)

When looking straight down, there is no privileged orientation — buildings and roads look the same from any angle. This gives us the **[D4](https://explore.albumentations.ai/transform/D4) dihedral symmetry group with 8 elements**:

- 4 rotations: 0°, 90°, 180°, 270°
- 4 reflections: horizontal flip, vertical flip, and flips along both diagonals

In Albumentations, this is captured by [`D4`](https://explore.albumentations.ai/transform/D4) (previously called `SquareSymmetry`):

```python
import albumentations as A

# D4 symmetry: all 8 elements of the dihedral group
# Perfect for satellite/aerial imagery TTA
tta_transforms = [
    A.Compose([]),                                    # Identity
    A.HorizontalFlip(p=1.0),                         # Horizontal flip
    A.VerticalFlip(p=1.0),                            # Vertical flip
    A.Compose([A.HorizontalFlip(p=1.0),
               A.VerticalFlip(p=1.0)]),               # 180° rotation
    A.Compose([A.Transpose(p=1.0)]),                  # Transpose
    A.Compose([A.Transpose(p=1.0),
               A.HorizontalFlip(p=1.0)]),             # 90° rotation
    A.Compose([A.Transpose(p=1.0),
               A.VerticalFlip(p=1.0)]),               # 270° rotation
    A.Compose([A.Transpose(p=1.0), A.HorizontalFlip(p=1.0),
               A.VerticalFlip(p=1.0)]),               # Transpose + 180°
]
```

### Medical Imaging

Medical images (histopathology, retinal scans, cell microscopy) often have similar symmetry properties to satellite imagery — there is no canonical "up" direction when looking at tissue under a microscope. **[D4](https://explore.albumentations.ai/transform/D4) symmetry** is standard for:

- Histopathology patches
- Cell segmentation
- Dermatoscopy images

For 3D medical volumes (CT, MRI), symmetries depend on the anatomy and acquisition protocol.

### Summary Table

| Domain | Geometric Symmetry | Group Size | Color TTA |
|---|---|---|---|
| Natural images (ImageNet) | [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip) | 2 | ✓ (small) |
| Satellite / aerial | [D4](https://explore.albumentations.ai/transform/D4) (all rotations + flips) | 8 | ✓ |
| Medical (microscopy) | [D4](https://explore.albumentations.ai/transform/D4) (all rotations + flips) | 8 | ✓ |
| Medical (CT/MRI slices) | Task-dependent | 2-8 | ✓ |
| Text / document images | None (usually) | 1 | Minimal |

## Resource Tradeoff

TTA trades **compute** for **accuracy**. Each additional view of the input requires a full forward pass through the network:

- **2 views** ([HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)) → **2× slower**
- **5 views** (FiveCrop) → **5× slower**
- **8 views** ([D4](https://explore.albumentations.ai/transform/D4) symmetry) → **8× slower**
- **10 views** (TenCrop) → **10× slower**

The accuracy gains follow a pattern of **diminishing returns**. The first few views provide the most benefit; adding more views continues to help but with progressively smaller improvements.

We can see this exact tradeoff curve mathematically when applying TTA to a standard ResNet18 ImageNet classifier:

![TTA Accuracy vs Number of Views](../../img/tta_accuracy_plot.webp)

Looking at the graph above:
- Adding just **1 extra view** (HorizontalFlip) captures an immediate +0.72% accuracy jump.
- Adding **3 more views** (FiveCrop) only captures an additional +0.79% accuracy bump.
- Pushing to **9 extra views** (TenCrop) costs nearly ~9x the inference time over baseline but only squeezed out a final +0.42% beyond FiveCrop.

For most applications, **[HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip) TTA** (2×) or **[D4](https://explore.albumentations.ai/transform/D4) TTA** (8× for applicable domains) offers the best accuracy-per-compute tradeoff.

## Usage in Kaggle Competitions

TTA is used in **virtually 100% of Kaggle competitions** involving computer vision. When competitors fight for every thousandth of a point on the leaderboard, TTA is one of the easiest and most reliable ways to squeeze out extra performance:

- It requires **zero retraining** — just modify the inference pipeline.
- It stacks with other ensemble techniques (model ensembles, snapshot ensembles).
- The improvement is **almost always positive** — it rarely hurts.

A typical Kaggle winner's inference pipeline looks like:

```
Final Prediction = Average(
    Model_1 × TTA_views +
    Model_2 × TTA_views +
    Model_3 × TTA_views +
    ...
)
```

Where each model is evaluated with multiple TTA views, and then all predictions are averaged. This "ensemble of ensembles" approach is standard in competition-winning solutions.

## Usage in Production

TTA is not just a competition trick — it is used in **production systems** where even small improvements in accuracy justify the additional compute cost:

- **Medical imaging** — A 0.5% improvement in tumor detection accuracy can save lives. The cost of running 8 forward passes instead of 1 is trivial compared to the cost of a missed diagnosis.
- **Satellite imagery analysis** — Monitoring deforestation, urban planning, and disaster response. Higher accuracy means better decisions.
- **Quality inspection** — Manufacturing defect detection where false negatives are expensive.
- **Autonomous driving** — Some perception pipelines use a form of TTA through multi-scale inference.

The decision is straightforward: if the cost of an error exceeds the cost of additional compute, TTA is worth using.

## Academic Misuse

> [!WARNING]
> TTA is sometimes used in academic papers in a questionable way that inflates apparent progress.

A common pattern in papers:

1. Train your model.
2. Evaluate with TTA and report the TTA-enhanced results.
3. Compare against previous papers that reported results **without TTA**.
4. Claim state-of-the-art.

This is misleading because the improvement may come entirely from TTA (which could be applied to any model) rather than from the proposed architectural or methodological contribution. The comparison is not apples-to-apples.

**Best practice**: always report results both **with and without TTA**, and compare against baselines under the same evaluation protocol. If you use TTA, your baselines should too.

## Practical Considerations

### When to Use TTA

✅ **Use TTA when:**

- Accuracy is more important than latency
- You have spare compute at inference time
- The task has clear geometric symmetries
- You're in a competition and every point matters

❌ **Skip TTA when:**

- Real-time inference is required (e.g., video processing at 30+ FPS)
- The compute budget is strictly constrained
- The base model is already very strong and gains are negligible
- The task has no meaningful symmetries

### Implementation Checklist

1. **Identify the symmetries** of your problem domain
2. **Use only transforms that were part of training** — TTA is most effective when the network has been trained with the same augmentations
3. **Start with [HorizontalFlip](https://explore.albumentations.ai/transform/HorizontalFlip)** — cheapest and most universal
4. **Average logits** rather than probabilities (slightly better in practice)
5. **For spatial tasks** (segmentation, detection), remember to apply **inverse transforms** to predictions
6. **Benchmark the improvement** — measure accuracy with and without TTA to confirm it helps for your specific model and dataset

## Full Benchmark Code

<details>
<summary>Complete TTA benchmark script (ResNet18 on ImageNet)</summary>

```python
"""Benchmark TTA strategies on ImageNet validation with ResNet18."""

import os
import time

import timm
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def create_base_transform():
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

# --- Load model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = timm.create_model("resnet18", pretrained=True).to(device).eval()

# --- Load image ---
image = np.array(Image.open("path/to/image.jpg").convert("RGB"))

# --- Baseline ---
transform = create_base_transform()
with torch.no_grad():
    logits = model(transform(image=image)["image"].unsqueeze(0).to(device))
    baseline_pred = logits.argmax(dim=1)

# --- HorizontalFlip TTA ---
flip = A.HorizontalFlip(p=1.0)
with torch.no_grad():
    logits_orig = model(transform(image=image)["image"].unsqueeze(0).to(device))
    logits_flip = model(transform(image=flip(image=image)["image"])["image"].unsqueeze(0).to(device))
    avg_logits = (logits_orig + logits_flip) / 2
    hflip_pred = avg_logits.argmax(dim=1)

# --- FiveCrop TTA ---
image_resized = A.SmallestMaxSize(max_size=256)(image=image)["image"]
h, w = image_resized.shape[:2]
crop_size = 224
boxes = [
    [0, 0, crop_size, crop_size],
    [w - crop_size, 0, w, crop_size],
    [0, h - crop_size, crop_size, h],
    [w - crop_size, h - crop_size, w, h],
    [(w - crop_size)//2, (h - crop_size)//2, (w + crop_size)//2, (h + crop_size)//2]
]
normalize = A.Compose([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

crops = []
for x_min, y_min, x_max, y_max in boxes:
    crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)(image=image_resized)["image"]
    crops.append(normalize(image=crop)["image"])

crop_tensors = torch.stack(crops).to(device)
with torch.no_grad():
    logits_5 = model(crop_tensors).mean(dim=0, keepdim=True)
    fivecrop_pred = logits_5.argmax(dim=1)

# --- TenCrop TTA ---
crops_10 = []
for x_min, y_min, x_max, y_max in boxes:
    crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)(image=image_resized)["image"]
    crops_10.append(normalize(image=crop)["image"])
    crops_10.append(normalize(image=flip(image=crop)["image"])["image"])

crop_tensors_10 = torch.stack(crops_10).to(device)
with torch.no_grad():
    logits_10 = model(crop_tensors_10).mean(dim=0, keepdim=True)
    tencrop_pred = logits_10.argmax(dim=1)
```

</details>

The complete runnable benchmark script is at [`scripts/tta_imagenet_benchmark.py`](../../scripts/tta-imagenet-benchmark.py).

## Where to Go Next?

-   **[Choosing Augmentations](../3-basic-usage/choosing-augmentations.md):** Learn which augmentations to use during training — TTA is most effective when matched to the training augmentation strategy.
-   **[Image Classification](../3-basic-usage/image-classification.md):** Set up classification training pipelines with Albumentations.
-   **[Semantic Segmentation](../3-basic-usage/semantic-segmentation.md):** Segmentation pipelines where TTA with equivariance shines.
-   **[Interactive Exploration](https://explore.albumentations.ai):** Visually experiment with transforms to understand which symmetries apply to your data.
