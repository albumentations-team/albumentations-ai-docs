# Reproducibility in Albumentations

Reproducibility is crucial for scientific experiments, debugging, and production deployments. This guide covers everything you need to know about creating reproducible augmentation pipelines in Albumentations.

## Quick Start

To make your augmentations reproducible, set the `seed` parameter in `Compose`:

```python
import albumentations as A

# This pipeline will produce the same augmentations
# every time it's instantiated with the same seed
transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=137)
```

## Key Concepts

### 1. Independent Random State

Albumentations maintains its own internal random state that is **completely independent** from global random seeds. This design choice ensures:

- Pipeline reproducibility is not affected by external code
- Multiple pipelines can coexist without interfering with each other
- Your augmentations remain consistent regardless of other random operations in your code

```python
import random
import numpy as np
import albumentations as A

# These global seeds DO NOT affect Albumentations
np.random.seed(137)
random.seed(137)

# Only the seed parameter in Compose controls reproducibility
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], seed=137)  # This is what matters
```

### 2. Seed Behavior

When you set a seed in `Compose`:

- **Two instances with the same seed produce identical sequences:**
  ```python
  transform1 = A.Compose([...], seed=137)
  transform2 = A.Compose([...], seed=137)
  # transform1 and transform2 will apply the same random parameters
  ```

- **Each call still produces random augmentations:**
  ```python
  transform = A.Compose([...], seed=137)
  # Different random augmentations for each call
  result1 = transform(image=image1)
  result2 = transform(image=image2)
  # But the sequence is reproducible when recreating the pipeline
  ```

- **No seed means truly random behavior:**
  ```python
  transform = A.Compose([...])  # seed=None by default
  # Different random sequence every time you create the pipeline
  ```

## Common Use Cases

### 1. Reproducible Training Experiments

```python
def create_train_transform(seed=None):
    """Create a training augmentation pipeline with optional seed."""
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.8
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ], seed=seed)

# For reproducible experiments
train_transform = create_train_transform(seed=137)

# For production (no fixed seed, different augmentations each run)
train_transform = create_train_transform(seed=None)
```

### 2. Debugging with Fixed Seeds

When debugging augmentation issues, use a fixed seed to ensure consistent behavior:

```python
# Debug mode - same augmentations every run
debug_transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=30,
        p=1.0  # Always apply for debugging
    ),
], seed=137)

# Test with the same image multiple times
for i in range(3):
    result = debug_transform(image=test_image)
    # Will produce the exact same augmented image each iteration
```

### 3. A/B Testing Augmentation Strategies

Compare different augmentation strategies with controlled randomness:

```python
# Strategy A with fixed seed
strategy_a = A.Compose([
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
], seed=100)

# Strategy B with the same seed for fair comparison
strategy_b = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.3),
], seed=100)

# Both will use the same random sequence for probability checks
```

### 4. Multi-Stage Pipelines

When using multiple Compose instances in sequence, each can have its own seed:

```python
# Stage 1: Geometric transforms
geometric = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
], seed=137)

# Stage 2: Color transforms
color = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
], seed=137)

# Apply stages sequentially
image = geometric(image=image)['image']
image = color(image=image)['image']
```

## Resetting Seeds for Existing Pipelines

You can reset the random seed of an existing pipeline without recreating it:

```python
import albumentations as A

# Create a pipeline
transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
], seed=137)

# Apply some augmentations
result1 = transform(image=image)

# Reset to a new seed
transform.set_random_seed(200)

# Now uses the new seed
result2 = transform(image=image)

# Reset to original seed
transform.set_random_seed(137)

# You can also set random state directly from generators
import numpy as np
import random

rng = np.random.default_rng(100)
py_rng = random.Random(100)
transform.set_random_state(rng, py_rng)
```

## DataLoader Workers and Reproducibility

**Key Concept**: In Albumentations, the augmentation sequence depends on BOTH the seed AND the number of workers. Using `seed=137` with `num_workers=4` produces different results than `seed=137` with `num_workers=8`. This is by design to maximize augmentation diversity in parallel processing.

### Automatic Worker Seed Handling

Albumentations automatically handles seed synchronization when used with PyTorch DataLoader workers:

```python
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

class MyDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image=image)['image']
        return image

    def __len__(self):
        return len(self.data)

# Create transform with seed
transform = A.Compose([
    A.RandomCrop(height=256, width=256),
    A.HorizontalFlip(p=0.5),
], seed=137)

dataset = MyDataset(images, transform=transform)

# Each worker gets a unique, reproducible seed automatically
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Multiple workers
    shuffle=True
)
```

### How Worker Seeds Work

1. **Base Seed**: When you set `seed=137` in Compose, this becomes the base seed
2. **Worker Differentiation**: Each worker automatically gets a unique seed based on:
   - Base seed (137)
   - PyTorch's worker-specific `torch.initial_seed()`
3. **Reproducibility**: The same worker ID always gets the same effective seed across runs
4. **Respawn Handling**: Seeds update correctly when workers are respawned

The effective seed formula:
```python
effective_seed = (base_seed + torch.initial_seed()) % (2**32)
```

### Important: Same Seed, Different num_workers = Different Augmentations

**Critical Note**: Using the same seed with different `num_workers` settings will produce different augmentation sequences:

```python
# Same seed=137, but different num_workers -> Different results!
transform = A.Compose([...], seed=137)

# With 1 worker
loader1 = DataLoader(dataset, num_workers=1)
# Worker 0 gets: effective_seed = 137 + torch_seed_0

# With 4 workers
loader2 = DataLoader(dataset, num_workers=4)
# Worker 0 gets: effective_seed = 137 + torch_seed_0
# Worker 1 gets: effective_seed = 137 + torch_seed_1
# Worker 2 gets: effective_seed = 137 + torch_seed_2
# Worker 3 gets: effective_seed = 137 + torch_seed_3
# Different data distribution across workers = different overall results!

# With 8 workers
loader3 = DataLoader(dataset, num_workers=8)
# 8 different effective seeds = yet another different result!
```

This is **by design** to ensure:
- Each worker produces unique augmentations (no duplicates)
- Maximum augmentation diversity in parallel processing
- Reproducibility when using the SAME num_workers configuration

**Key insight**: The augmentation sequence depends on BOTH the seed AND num_workers. To get identical results, you must use the same seed AND the same num_workers.

### Manual Worker Seed Management

If you need custom worker seed logic:

```python
def worker_init_fn(worker_id):
    # Custom seed logic
    worker_seed = torch.initial_seed() % 2**32
    # The transform will automatically use this for differentiation

dataloader = DataLoader(
    dataset,
    num_workers=4,
    worker_init_fn=worker_init_fn
)
```

### Single Process vs Multi-Process

```python
# Single process (num_workers=0)
# Uses base seed directly
transform = A.Compose([...], seed=137)
loader = DataLoader(dataset, num_workers=0)
# Always produces the same sequence

# Multi-process (num_workers>0)
# Each worker gets unique seed automatically
loader = DataLoader(dataset, num_workers=4)
# Each worker produces different sequences
# But sequences are reproducible across runs
```

### Making Augmentations Identical Across Different num_workers

**Important**: By design, different `num_workers` values produce different augmentation sequences, even with the same seed. This is because each worker gets a unique effective seed. If you need identical augmentations regardless of `num_workers` (unlikely but possible use case), here are some workarounds:

**Note**: When using `persistent_workers=True`, the difference becomes more pronounced as the worker seed state may not reset properly between epochs.

```python
# Option 1: Force same seed for all workers (ignores worker ID)
class IdenticalAugmentDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Create fixed random generators shared across all workers
        self.rng = np.random.default_rng(137)
        self.py_rng = random.Random(137)
        self.transform = A.Compose([
            A.RandomCrop(height=256, width=256),
            A.HorizontalFlip(p=0.5),
        ])  # No seed here!

    def __getitem__(self, idx):
        # Force the same random state regardless of worker
        self.transform.set_random_state(self.rng, self.py_rng)
        image = self.data[idx]
        return self.transform(image=image)['image']

# Now num_workers=4 and num_workers=8 produce identical sequences

# Option 2: Use a fixed seed that ignores worker differentiation
def worker_init_fn(worker_id):
    # Override the automatic worker seed differentiation
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset = worker_info.dataset
        # Use the same seed for all workers (not recommended for training!)
        dataset.transform = A.Compose([
            A.RandomCrop(height=256, width=256),
            A.HorizontalFlip(p=0.5),
        ], seed=137)  # Same seed, ignoring worker_id
```

**Warning**: Making augmentations identical across different `num_workers` defeats the purpose of parallel data loading and reduces augmentation diversity. This is typically only useful for debugging or specific reproducibility requirements.

This behavior is discussed in [Albumentations Issue #81](https://github.com/albumentations-team/AlbumentationsX/issues/81).

## Custom Transforms and Reproducibility

When creating custom transforms, use the provided random generators to maintain reproducibility:

```python
from albumentations.core.transforms_interface import DualTransform

class MyCustomTransform(DualTransform):
    def get_params_dependent_on_data(self, params, data):
        # CORRECT: Use self.py_random for Python's random operations
        random_value = self.py_random.uniform(0, 1)

        # CORRECT: Use self.random_generator for NumPy operations
        random_array = self.random_generator.uniform(0, 1, size=(3, 3))

        # WRONG: Don't use global random functions
        # bad_value = random.random()  # This ignores the seed!
        # bad_array = np.random.rand(3, 3)  # This also ignores the seed!

        return {"value": random_value, "array": random_array}
```

See the [Creating Custom Transforms Guide](./creating-custom-transforms.md#reproducibility-and-random-number-generation) for more details.

## Saving and Loading Pipelines

For perfect reproducibility across different environments or time, save your pipeline configuration:

```python
# Save pipeline configuration
A.save(transform, 'augmentation_pipeline.json')

# Load the exact same pipeline later
transform = A.load('augmentation_pipeline.json')
```

Note: The loaded pipeline will have the same seed as the original. See the [Serialization Guide](./serialization.md) for more details.

## Tracking Applied Augmentations

To debug or analyze which augmentations were actually applied, use `save_applied_params`:

```python
transform = A.Compose([
    A.RandomCrop(256, 256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
], save_applied_params=True, seed=137)

result = transform(image=image)
print(transform.applied_transforms)
# Shows exactly which transforms were applied and their parameters
```

## Best Practices

1. **Development vs Production:**
   - Use fixed seeds during development and debugging
   - Remove seeds (or use different seeds per epoch) in production training
   - Always use fixed seeds for validation/test transforms if you need comparable results

2. **Experiment Tracking:**
   - Log the seed value in your experiment tracking system
   - Save the complete pipeline configuration using `A.save()`
   - Document the Albumentations version used
   - Track `num_workers` setting as it affects augmentation sequences

3. **Testing:**
   - Unit tests should always use fixed seeds
   - Integration tests may use random seeds to test robustness
   - Create separate test cases for both scenarios
   - Test with both single and multi-worker configurations

4. **Distributed Training:**
   - Albumentations automatically handles worker differentiation
   - Each worker gets a unique, reproducible seed based on `base_seed + torch.initial_seed()`
   - No need for manual `seed = base_seed + worker_id` logic
   - Seeds are automatically updated on worker respawn

5. **DataLoader Configuration:**
   - Be aware that changing `num_workers` changes augmentation sequences
   - Document your `num_workers` setting for reproducibility
   - Use consistent `num_workers` across experiments for comparable results
   - Avoid `persistent_workers=True` if exact reproducibility is critical (see known issue below)

## Common Pitfalls

### ❌ Don't rely on global seeds

```python
# This WILL NOT make Albumentations reproducible
np.random.seed(137)
random.seed(137)

transform = A.Compose([...])  # Still random!
```

### ❌ Don't forget that each Compose call is still random

```python
transform = A.Compose([...], seed=137)

# These will be different (but reproducible sequence)
aug1 = transform(image=img)
aug2 = transform(image=img)  # Different augmentation!
```

### ✅ Do create new instances for identical augmentations

```python
# If you need the exact same augmentation
transform1 = A.Compose([...], seed=137)
transform2 = A.Compose([...], seed=137)

# Now these will be identical
aug1 = transform1(image=img)
aug2 = transform2(image=img)  # Same augmentation!
```

## Summary

This guide covered:
- Setting and resetting seeds for reproducible augmentations
- Automatic worker seed handling in PyTorch DataLoaders
- How different `num_workers` settings affect augmentation sequences
- Best practices for reproducible experiments
- Common pitfalls and how to avoid them

## Related Topics

- [Pipelines and Compose](../2-core-concepts/pipelines.md) - Understanding pipeline configuration
- [Probabilities](../2-core-concepts/probabilities.md) - How probabilities interact with seeds
- [Creating Custom Transforms](./creating-custom-transforms.md#reproducibility-and-random-number-generation) - Making custom transforms reproducible
- [Serialization](./serialization.md) - Saving and loading reproducible pipelines
