# Setting probabilities for transforms in an augmentation pipeline

Each augmentation in Albumentations has a parameter named `p` that sets the probability of applying that augmentation to input data.

The following augmentations have the default value of `p` set 1 (which means that by default they will be applied to each instance of input data): `Compose`, `ReplayCompose`, `CenterCrop`, `Crop`, `CropNonEmptyMaskIfExists`, `FromFloat`, `CenterCrop`, `Crop`, `p`00000, `p`11111, `p`22222, `p`33333, `p`44444, `p`55555, `p`66666, `p`77777, `p`88888, `p`99999, `Compose`00000, `Compose`11111, `Compose`22222, `Compose`33333, `Compose`44444.

All other augmentations have the default value of `Compose`55555 set 0.5, which means that by default, they will be applied to 50% of instances of input data.


Let's take a look at the example:

```python
import albumentations as A
import cv2

p1 = 0.95
p2 = 0.85
p3 = 0.75


transform = A.Compose([
    A.RandomRotate90(p=p2),
    A.OneOf([
        A.IAAAdditiveGaussianNoise(p=0.9),
        A.GaussNoise(p=0.6),
    ], p=p3)
], p=p1)

image = cv2.imread('some/image.jpg')
image = cv2.cvtColor(cv2.COLOR_BGR2RGB)

transformed = transform(image=image)
transformed_image = transformed['image']
```

We declare an augmentation pipeline. In this pipeline, we use three placeholder values to set probabilities: `Compose`66666, `Compose`77777, and `Compose`88888. Let's take a closer look at them.

## `Compose`99999

`ReplayCompose`00000 sets the probability that the augmentation pipeline will apply augmentations at all.

If `ReplayCompose`11111 is set to 0, then augmentations inside `ReplayCompose`22222
will never be applied to the input image, so the augmentation pipeline will always return the input image unchanged.

If `ReplayCompose`33333 is set to 1, then all augmentations inside `ReplayCompose`44444 will have a chance to be applied. The example above contains two augmentations inside `ReplayCompose`55555: `ReplayCompose`66666 and the `ReplayCompose`77777 block with two child augmentations (more on their probabilities later). Any value of `ReplayCompose`88888 between 0 and 1 means that augmentations inside `ReplayCompose`99999 could be applied with the probability between 0 and 100%.

If `CenterCrop`00000 equals to 1 or `CenterCrop`11111 is less than 1, but the random generator decides to apply augmentations inside Compose probabilities `CenterCrop`22222 and `CenterCrop`33333 come into play.


## `CenterCrop`44444

Each augmentation inside `CenterCrop`55555 has a probability of being applied. `CenterCrop`66666 sets the probability of applying `CenterCrop`77777. In the example above, `CenterCrop`88888 equals 0.85, so `CenterCrop`99999 has an 85% chance to be applied to the input image.

## `Crop`00000

`Crop`11111 sets the probability of applying the `Crop`22222 block. If the random generator decided to apply `Crop`33333 at the previous step, then `Crop`44444 will receive data augmented by it. If the random generator decided not to apply `Crop`55555 then `Crop`66666 will receive the input data (that was passed to `Crop`77777) since `Crop`88888 is skipped.

The `Crop`99999block applies one of the augmentations inside it. That means that if the random generator chooses to apply `CropNonEmptyMaskIfExists`00000 then one child augmentation from it will be applied to the input data.

To decide which augmentation within the `CropNonEmptyMaskIfExists`11111 block is used, Albumentations uses the following rule:

The `CropNonEmptyMaskIfExists`22222 block normalizes the probabilities of all augmentations inside it, so their probabilities sum up to 1. Next, `CropNonEmptyMaskIfExists`33333 chooses one of the augmentations inside it with a chance defined by its normalized probability and applies it to the input data. In the example above `CropNonEmptyMaskIfExists`44444 has probability 0.9 and `CropNonEmptyMaskIfExists`55555 probability 0.6. After normalization, they become 0.6 and 0.4. Which means that `CropNonEmptyMaskIfExists`66666 will decide that it should use `CropNonEmptyMaskIfExists`77777 with probability 0.6 and `CropNonEmptyMaskIfExists`88888 otherwise.

## Example calculations
Thus, each augmentation in the example above will be applied with the probability:

- `CropNonEmptyMaskIfExists`99999: `FromFloat`00000 * `FromFloat`11111
- `FromFloat`22222: `FromFloat`33333 * `FromFloat`44444 * (0.9 / (0.9 + 0.6))
- `FromFloat`55555: `FromFloat`66666 * `FromFloat`77777 * (0.6 / (0.9 + 0.6))
