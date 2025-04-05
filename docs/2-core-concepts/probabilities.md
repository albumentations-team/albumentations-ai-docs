# Setting probabilities for transforms in an augmentation pipeline

Each augmentation in Albumentations has a parameter named `p` that sets the probability of applying that augmentation to input data. Setting `p=1` means the transform will always be considered for application, while `p=0` means it will never be considered. A value between 0 and 1 represents the chance it will be considered.

Some transforms default to `p=1`, while others default to `p=0.5`. Since default values can vary, it is recommended to explicitly set the `p` value for each transform in your pipeline to ensure clarity and avoid unexpected behavior.

Let's take a look at an example:

```python
import albumentations as A
import cv2
import numpy as np # Assuming image is a numpy array

# Define probabilities
prob_pipeline = 0.95
prob_rotate = 0.85
prob_oneof_noise = 0.75

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=prob_rotate), # Has an 85% chance to be considered
    A.OneOf([
        A.GaussNoise(p=0.9),         # If OneOf is chosen, GaussNoise has a 90% chance within the OneOf block
        A.ISONoise(p=0.7),           # If OneOf is chosen, ISONoise has a 70% chance within the OneOf block
    ], p=prob_oneof_noise)            # The OneOf block has a 75% chance to be considered
], p=prob_pipeline,                  # The entire pipeline has a 95% chance to be applied
  seed=137)                         # Added seed for reproducibility

# Load an example image (replace with your image path)
# image = cv2.imread('some/image.jpg')
# image = cv2.cvtColor(cv2.COLOR_BGR2RGB)
# For demonstration, let's create a dummy image
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


transformed = transform(image=image)
transformed_image = transformed['image']

print("Transformation applied:", not np.array_equal(image, transformed_image))
```

We declare an augmentation pipeline using `Compose`. The probability `p` for `Compose` itself (`prob_pipeline`) determines if *any* augmentations within it are applied.

## Pipeline Probability (`prob_pipeline`)

`prob_pipeline` (set to 0.95 here) is the overall probability that the `Compose` block executes.
- If `prob_pipeline` is 0, the pipeline never runs, and the input is always returned unchanged.
- If `prob_pipeline` is 1, the pipeline always runs, and the augmentations *inside* it get a chance to be applied based on their own probabilities.
- If `0 < prob_pipeline < 1`, the pipeline runs with that specific probability.

Assuming the pipeline runs (which happens 95% of the time in our example), the probabilities of the inner transforms (`prob_rotate`, `prob_oneof_noise`) come into play.

## Individual Transform Probability (`prob_rotate`)

`prob_rotate` (set to 0.85) is the probability that [`RandomRotate90`](https://explore.albumentations.ai/transform/RandomRotate90) is applied, *given* the pipeline runs. So, [`RandomRotate90`](https://explore.albumentations.ai/transform/RandomRotate90) has an 85% chance of being applied each time the `Compose` block is executed.

## `OneOf` Block Probability (`prob_oneof_noise`)

`prob_oneof_noise` (set to 0.75) sets the probability that the `OneOf` block is applied, *given* the pipeline runs.

If the `OneOf` block is selected (75% chance when the pipeline runs), it will execute *exactly one* of the transforms defined within it ([`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise) or [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise)).

### Probabilities within `OneOf`

To decide which transform inside `OneOf` is used, Albumentations normalizes the probabilities of the inner transforms so they sum to 1.
- [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise) has `p=0.9`
- [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise) has `p=0.7`
- The sum is `0.9 + 0.7 = 1.6`.
- Normalized probability for [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise): `0.9 / 1.6 = 0.5625` (or 56.25%)
- Normalized probability for [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise): `0.7 / 1.6 = 0.4375` (or 43.75%)

So, if the `OneOf` block runs, [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise) will be applied 56.25% of the time, and [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise) will be applied 43.75% of the time.

## Overall Probabilities

The actual probability of each specific transform being applied to the original image is:
- [`RandomRotate90`](https://explore.albumentations.ai/transform/RandomRotate90): `prob_pipeline * prob_rotate` = `0.95 * 0.85` = `0.8075` (80.75%)
- [`GaussNoise`](https://explore.albumentations.ai/transform/GaussNoise): `prob_pipeline * prob_oneof_noise * (0.9 / (0.9 + 0.7))` = `0.95 * 0.75 * 0.5625` = `0.40078125` (approx 40.08%)
- [`ISONoise`](https://explore.albumentations.ai/transform/ISONoise): `prob_pipeline * prob_oneof_noise * (0.7 / (0.9 + 0.7))` = `0.95 * 0.75 * 0.4375` = `0.3115234375` (approx 31.15%)

## When `p=1` Might Not Change the Image

Generally, if a transform is applied (i.e., `p=1` or the random chance based on `p` succeeds), you can expect the output image to be different from the input. However, there are some specific corner cases where a transform runs but might not alter the image:

1.  **Transforms Sampling from a Group Including Identity:** Some transforms randomly select an operation from a predefined set, and one of those operations might be the "identity" (do nothing) operation.
    *   Example: [`RandomRotate90(p=1)`](https://explore.albumentations.ai/transform/RandomRotate90) randomly chooses a rotation of 0, 90, 180, or 270 degrees. There's a 1 in 4 chance it selects 0 degrees, leaving the image unchanged.
    *   Other examples include [`D4`](https://explore.albumentations.ai/transform/D4) (symmetries of a square) and [`RandomGridShuffle`](https://explore.albumentations.ai/transform/RandomGridShuffle) (which might randomly shuffle grid cells back to their original positions).

2.  **Geometric Transforms with Identity Parameters:** Many geometric transforms sample parameters (like rotation angle, scaling factor, translation distance) randomly within a specified range. If the randomly chosen parameters happen to correspond to the identity transformation (e.g., rotate by 0 degrees, scale by 1, translate by 0 pixels), the image won't change, even if `p=1`.
    *   Example: [`Affine(rotate=0, scale=1, translate_px=0, p=1)`](https://explore.albumentations.ai/transform/Affine) will always apply the identity transform.
    *   Example: [`Affine(rotate=(-10, 10), p=1)`](https://explore.albumentations.ai/transform/Affine) might randomly sample a rotation angle of exactly 0, resulting in no change from rotation (though scaling or translation might still occur if enabled).
    *   Example: [`ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=1)`](https://explore.albumentations.ai/transform/ShiftScaleRotate) could randomly sample shift=0, scale=1, and rotate=0 simultaneously.

So, while `p` controls the probability of a transform *being executed*, the specific internal logic and random parameter sampling of the transform determine if that execution *results* in a visually modified image.
