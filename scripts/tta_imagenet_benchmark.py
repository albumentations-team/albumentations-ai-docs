"""Benchmark Test Time Augmentation (TTA) strategies on ImageNet validation.

Evaluates ResNet18 (timm) on ImageNet validation with:
1. Baseline (center crop)
2. HorizontalFlip TTA
3. FiveCrop TTA
4. TenCrop TTA

Usage:
    python scripts/tta_imagenet_benchmark.py --data-dir ~/data/imagenet/val
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Standard ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# URL sourcing the canonical ImageNet validation ground truth
# Maps each of the 50,000 val images (sorted by filename) to its synset ID
SYNSET_LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)


def get_device() -> torch.device:
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_imagenet_synsets() -> list[str]:
    """Return the 1000 ImageNet synsets in the canonical sorted order (matching timm/torchvision class indices)."""
    # timm models use the same class ordering as torchvision: synsets sorted alphabetically
    # We can extract this from the model's pretrained config
    model = timm.create_model("resnet18", pretrained=False)
    # timm stores imagenet class info
    try:
        from timm.data import ImageNetInfo

        info = ImageNetInfo()
        # Build sorted synset list
        synsets = []
        for idx in range(1000):
            label_info = info.index_to_description(idx)
            synsets.append(info.index_to_synset(idx) if hasattr(info, "index_to_synset") else None)
        if synsets[0] is not None:
            return synsets
    except Exception:
        pass

    # Fallback: get synsets from the model config or download standard list
    return _download_synset_list()


def _download_synset_list() -> list[str]:
    """Download the canonical list of 1000 ImageNet synsets in sorted order."""
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    # Actually, we need synset IDs. Use a different source.
    # The standard approach: get all 1000 synset IDs from ILSVRC
    # These are available from many sources. We'll extract them from timm directly.
    model = timm.create_model("resnet18", pretrained=False, num_classes=1000)
    # timm stores the label mapping in the default_cfg
    # For a more reliable approach, use the known sorted synset list
    synset_url = (
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    # Actually, what we really need is synset WordNet IDs (wnids), not human-readable names
    # The authoritative source is the ILSVRC devkit, but we can reconstruct from various sources
    # For this benchmark, let's use a simpler approach: download the wnid list
    return []


def load_val_labels(data_dir: str, cache_dir: str | None = None) -> list[int]:
    """Load ImageNet validation ground truth labels.

    Downloads the synset labels for each of the 50,000 validation images,
    then maps them to integer class indices (0-999) using the canonical
    alphabetical synset ordering that timm and torchvision use.

    Args:
        data_dir: Path to the ImageNet validation directory.
        cache_dir: Directory to cache downloaded label files. Defaults to data_dir parent.

    Returns:
        List of 50,000 integer class labels (0-999).
    """
    if cache_dir is None:
        cache_dir = str(Path(data_dir).parent)

    cache_file = os.path.join(cache_dir, "val_labels.json")

    # Return cached labels if available
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            labels = json.load(f)
        print(f"Loaded cached labels from {cache_file} ({len(labels)} labels)")
        return labels

    print("Downloading ImageNet validation labels...")

    # Step 1: Download per-image synset labels (50,000 lines)
    print(f"  Fetching synset labels from {SYNSET_LABELS_URL}")
    with urllib.request.urlopen(SYNSET_LABELS_URL) as response:
        synset_per_image = response.read().decode("utf-8").strip().split("\n")

    assert len(synset_per_image) == 50000, f"Expected 50000 synset labels, got {len(synset_per_image)}"

    # Step 2: Build the synset-to-index mapping
    # The canonical ordering is: sort all unique synsets alphabetically, assign 0-999
    unique_synsets = sorted(set(synset_per_image))
    assert len(unique_synsets) == 1000, f"Expected 1000 unique synsets, got {len(unique_synsets)}"
    synset_to_idx = {s: i for i, s in enumerate(unique_synsets)}

    # Step 3: Map each image to its class index
    labels = [synset_to_idx[s] for s in synset_per_image]

    # Cache for future runs
    with open(cache_file, "w") as f:
        json.dump(labels, f)
    print(f"  Cached labels to {cache_file}")

    return labels


def build_image_list(data_dir: str) -> list[str]:
    """Get sorted list of validation image paths."""
    extensions = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}
    images = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if os.path.splitext(f)[1] in extensions
    )
    return images


def create_base_transform():
    """Standard ImageNet validation transform: resize 256 → center crop 224."""
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def create_resize_transform():
    """Resize to 256 (for FiveCrop/TenCrop) so that DataLoader can collate them."""
    return A.Compose([
        A.SmallestMaxSize(max_size=256),
    ])


class ImageClassificationDataset(Dataset):
    """Simple Dataset for image classification."""

    def __init__(self, image_paths: list[str], labels: list[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | np.ndarray, int]:
        img_path = self.image_paths[idx]
        try:
            # We load as PIL and convert to numpy array for albumentations
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"  Warning: Could not load {img_path}: {e}")
            # Fallback to a zero array if image is corrupted
            img = np.zeros((256, 256, 3), dtype=np.uint8)

        # Apply transform if provided, but some methods (FiveCrop/TenCrop)
        # do their own multi-crop logic inside the eval loop, so transform might be None
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img, self.labels[idx]

@torch.no_grad()
def evaluate_baseline(
    model: torch.nn.Module,
    image_paths: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Evaluate with standard center crop (no TTA)."""
    dataset = ImageClassificationDataset(image_paths, labels, transform=create_base_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    for images, targets in tqdm(dataloader, desc="Baseline"):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        _, pred_top5 = logits.topk(5, dim=1)

        correct_top1 += (pred_top5[:, 0] == targets).sum().item()
        correct_top5 += (pred_top5 == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    elapsed = time.time() - start_time
    return {
        "method": "Baseline (Center Crop)",
        "top1_accuracy": correct_top1 / total * 100,
        "top5_accuracy": correct_top5 / total * 100,
        "total_images": total,
        "time_seconds": elapsed,
        "images_per_second": total / elapsed,
    }


@torch.no_grad()
def evaluate_horizontal_flip_tta(
    model: torch.nn.Module,
    image_paths: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    """Evaluate with HorizontalFlip TTA: average predictions of original + flipped."""
    dataset = ImageClassificationDataset(image_paths, labels, transform=None)
    transform = create_base_transform()
    flip = A.HorizontalFlip(p=1.0)  # deterministic flip
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    # Iterate over dataset in chunks of batch_size
    dataset_size = len(dataset)
    for start_idx in tqdm(range(0, dataset_size, batch_size), desc="HorizontalFlip TTA"):
        end_idx = min(start_idx + batch_size, dataset_size)

        batch_orig = []
        batch_flip = []
        batch_targets = []

        for i in range(start_idx, end_idx):
            img, target_val = dataset[i]
            batch_orig.append(transform(image=img)["image"])
            batch_flip.append(transform(image=flip(image=img)["image"])["image"])
            batch_targets.append(target_val)

        orig_tensor = torch.stack(batch_orig).to(device)
        flip_tensor = torch.stack(batch_flip).to(device)
        targets_tensor = torch.tensor(batch_targets, device=device)

        # Average logits from both views
        logits = (model(orig_tensor) + model(flip_tensor)) / 2.0

        _, pred_top5 = logits.topk(5, dim=1)
        correct_top1 += (pred_top5[:, 0] == targets_tensor).sum().item()
        correct_top5 += (pred_top5 == targets_tensor.unsqueeze(1)).any(dim=1).sum().item()
        total += targets_tensor.size(0)

    elapsed = time.time() - start_time
    return {
        "method": "HorizontalFlip TTA",
        "top1_accuracy": correct_top1 / total * 100,
        "top5_accuracy": correct_top5 / total * 100,
        "total_images": total,
        "time_seconds": elapsed,
        "images_per_second": total / elapsed,
    }


@torch.no_grad()
def evaluate_five_crop_tta(
    model: torch.nn.Module,
    image_paths: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Evaluate with FiveCrop TTA: average predictions over 5 crops (4 corners + center)."""
    dataset = ImageClassificationDataset(image_paths, labels, transform=None)
    normalize = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    resize = A.SmallestMaxSize(max_size=256)
    crop_size = 224
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    for i in tqdm(range(len(dataset)), desc="FiveCrop TTA"):
        img, target_val = dataset[i]

        img_resized = resize(image=img)["image"]
        h, w = img_resized.shape[:2]
        boxes = [
            [0, 0, crop_size, crop_size],
            [w - crop_size, 0, w, crop_size],
            [0, h - crop_size, crop_size, h],
            [w - crop_size, h - crop_size, w, h],
            [(w - crop_size)//2, (h - crop_size)//2, (w + crop_size)//2, (h + crop_size)//2]
        ]

        crops = []
        for x_min, y_min, x_max, y_max in boxes:
            crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)(image=img_resized)["image"]
            crops.append(normalize(image=crop)["image"])

        crop_tensors = torch.stack(crops).to(device)
        logits = model(crop_tensors)  # [5, 1000]
        avg_logits = logits.mean(dim=0, keepdim=True)  # [1, 1000]

        _, pred_top5 = avg_logits.topk(5, dim=1)
        correct_top1 += (pred_top5[0, 0].item() == target_val)
        correct_top5 += (target_val in pred_top5[0].tolist())
        total += 1

    elapsed = time.time() - start_time
    return {
        "method": "FiveCrop TTA",
        "top1_accuracy": correct_top1 / total * 100,
        "top5_accuracy": correct_top5 / total * 100,
        "total_images": total,
        "time_seconds": elapsed,
        "images_per_second": total / elapsed,
    }


@torch.no_grad()
def evaluate_ten_crop_tta(
    model: torch.nn.Module,
    image_paths: list[str],
    labels: list[int],
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    """Evaluate with TenCrop TTA: FiveCrop + their horizontal flips."""
    dataset = ImageClassificationDataset(image_paths, labels, transform=None)
    normalize = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    resize = A.SmallestMaxSize(max_size=256)
    hflip = A.HorizontalFlip(p=1.0)
    crop_size = 224
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0
    start_time = time.time()

    for i in tqdm(range(len(dataset)), desc="TenCrop TTA"):
        img, target_val = dataset[i]

        img_resized = resize(image=img)["image"]
        h, w = img_resized.shape[:2]
        boxes = [
            [0, 0, crop_size, crop_size],
            [w - crop_size, 0, w, crop_size],
            [0, h - crop_size, crop_size, h],
            [w - crop_size, h - crop_size, w, h],
            [(w - crop_size)//2, (h - crop_size)//2, (w + crop_size)//2, (h + crop_size)//2]
        ]

        crops = []
        for x_min, y_min, x_max, y_max in boxes:
            crop = A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)(image=img_resized)["image"]
            crops.append(normalize(image=crop)["image"])
            crops.append(normalize(image=hflip(image=crop)["image"])["image"])

        crop_tensors = torch.stack(crops).to(device)
        logits = model(crop_tensors)  # [10, 1000]
        avg_logits = logits.mean(dim=0, keepdim=True)  # [1, 1000]

        _, pred_top5 = avg_logits.topk(5, dim=1)
        correct_top1 += (pred_top5[0, 0].item() == target_val)
        correct_top5 += (target_val in pred_top5[0].tolist())
        total += 1

    elapsed = time.time() - start_time
    return {
        "method": "TenCrop TTA",
        "top1_accuracy": correct_top1 / total * 100,
        "top5_accuracy": correct_top5 / total * 100,
        "total_images": total,
        "time_seconds": elapsed,
        "images_per_second": total / elapsed,
    }


def print_results(results: list[dict]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 80)
    print("TEST TIME AUGMENTATION BENCHMARK RESULTS")
    print("Model: ResNet18 (timm, pretrained on ImageNet)")
    print("=" * 80)
    print(f"{'Method':<25} {'Top-1 Acc':>10} {'Top-5 Acc':>10} {'Time (s)':>10} {'Img/s':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['method']:<25} "
            f"{r['top1_accuracy']:>9.2f}% "
            f"{r['top5_accuracy']:>9.2f}% "
            f"{r['time_seconds']:>9.1f}s "
            f"{r['images_per_second']:>9.1f}"
        )
    print("=" * 80)

    # Show improvement over baseline
    if len(results) > 1:
        baseline = results[0]
        print("\nImprovement over baseline:")
        for r in results[1:]:
            top1_delta = r["top1_accuracy"] - baseline["top1_accuracy"]
            top5_delta = r["top5_accuracy"] - baseline["top5_accuracy"]
            slowdown = r["time_seconds"] / baseline["time_seconds"]
            print(
                f"  {r['method']:<25} "
                f"Top-1: {top1_delta:>+.2f}%  "
                f"Top-5: {top5_delta:>+.2f}%  "
                f"Slowdown: {slowdown:.1f}x"
            )


def main() -> None:
    """Run TTA benchmark."""
    parser = argparse.ArgumentParser(description="TTA Benchmark on ImageNet Validation")
    parser.add_argument("--data-dir", type=str, default=os.path.expanduser("~/data/imagenet/val"),
                        help="Path to ImageNet validation images (flat directory)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for baseline/flip evaluation")
    parser.add_argument("--num-images", type=int, default=None, help="Number of images to evaluate (default: all)")
    parser.add_argument("--skip-five-crop", action="store_true", help="Skip FiveCrop evaluation")
    parser.add_argument("--skip-ten-crop", action="store_true", help="Skip TenCrop evaluation")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # Load model
    print("Loading ResNet18 model...")
    model = timm.create_model("resnet18", pretrained=True)
    model = model.to(device)
    model.eval()

    # Load data
    print(f"Loading images from {args.data_dir}...")
    image_paths = build_image_list(args.data_dir)
    labels = load_val_labels(args.data_dir)

    if args.num_images is not None:
        image_paths = image_paths[:args.num_images]
        labels = labels[:args.num_images]

    print(f"Evaluating {len(image_paths)} images\n")

    results = []

    # 1. Baseline
    print("=" * 40)
    print("Running: Baseline (Center Crop)")
    print("=" * 40)
    results.append(evaluate_baseline(model, image_paths, labels, device, args.batch_size))
    print(f"  → Top-1: {results[-1]['top1_accuracy']:.2f}%  Top-5: {results[-1]['top5_accuracy']:.2f}%")

    # 2. HorizontalFlip TTA
    print("\n" + "=" * 40)
    print("Running: HorizontalFlip TTA")
    print("=" * 40)
    results.append(evaluate_horizontal_flip_tta(model, image_paths, labels, device, args.batch_size))
    print(f"  → Top-1: {results[-1]['top1_accuracy']:.2f}%  Top-5: {results[-1]['top5_accuracy']:.2f}%")

    # 3. FiveCrop TTA
    if not args.skip_five_crop:
        print("\n" + "=" * 40)
        print("Running: FiveCrop TTA")
        print("=" * 40)
        results.append(evaluate_five_crop_tta(model, image_paths, labels, device))
        print(f"  → Top-1: {results[-1]['top1_accuracy']:.2f}%  Top-5: {results[-1]['top5_accuracy']:.2f}%")

    # 4. TenCrop TTA
    if not args.skip_ten_crop:
        print("\n" + "=" * 40)
        print("Running: TenCrop TTA")
        print("=" * 40)
        results.append(evaluate_ten_crop_tta(model, image_paths, labels, device))
        print(f"  → Top-1: {results[-1]['top1_accuracy']:.2f}%  Top-5: {results[-1]['top5_accuracy']:.2f}%")

    # Print summary
    print_results(results)

    # Save results to JSON
    results_file = os.path.join(os.path.dirname(__file__), "tta_benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
