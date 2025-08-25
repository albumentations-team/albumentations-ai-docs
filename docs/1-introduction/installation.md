# Installation

Albumentations requires Python 3.9 or higher. We recommend using the latest stable Python version.

## Installation Methods

### AlbumentationsX (Recommended - Drop-in Replacement)

**AlbumentationsX** is the next-generation successor to Albumentations, offering:
- 🚀 100% API compatibility - no code changes required
- ⚡ Improved performance and bug fixes
- 🔧 Active maintenance and new features
- 📊 Better support for production environments

AlbumentationsX is dual-licensed (AGPL/Commercial). For more information about licensing, see our [License Guide](../license.md).

```bash
pip install -U albumentationsx
```

Your existing code continues to work without any changes:
```python
# Same import - no changes needed!
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])
```





### From GitHub (Latest Development Version)

For AlbumentationsX:
```bash
pip install -U git+https://github.com/albumentations-team/AlbumentationsX
```

**Note:** Installing from the `main` branch might give you newer features but could potentially be less stable than official releases.

## Handling OpenCV Dependencies

Both Albumentations and AlbumentationsX rely heavily on OpenCV.

*   **Default:** By default, they depend on `opencv-python-headless`. This version is chosen because it avoids installing GUI-related dependencies, making it suitable for server environments and containers where graphical interfaces are not needed.
*   **Using Your Existing OpenCV:** If you already have a different OpenCV distribution installed (like `opencv-python`, `opencv-contrib-python`, or `opencv-contrib-python-headless`), pip should automatically detect and use it.
*   **Forcing Source Build (Advanced):** If you need to force pip to build from source and use a *specific*, pre-existing OpenCV installation (perhaps compiled manually), you can use the `--no-binary` flag:
    ```bash
    # For AlbumentationsX
    pip install -U albumentationsx --no-binary albumentationsx
    ```
    In most standard cases, this flag is **not** required.

## Verify Installation

After installation, you can verify it by running:

```bash
python -c "import albumentations as A; print(A.__version__)"
```

This should print the installed version number.

## Telemetry in AlbumentationsX

AlbumentationsX includes anonymous usage telemetry to help improve the library. This can be disabled by:

Setting an environment variable:
```bash
export ALBUMENTATIONS_NO_TELEMETRY=1
```

Or per-pipeline:
```python
transform = A.Compose([...], telemetry=False)
```

Learn more in our [License Guide](../license.md#telemetry).

## Where to Go Next?

Now that you have Albumentations installed, here are some logical next steps:

-   **[Understand Core Concepts](../2-core-concepts/index.md):** Learn about transforms, pipelines, targets, and probabilities - the fundamental building blocks of Albumentations.
-   **[See Basic Usage Examples](../3-basic-usage/index.md):** Explore how to apply augmentations for common computer vision tasks.
-   **[Explore Transforms](https://explore.albumentations.ai):** Visually experiment with different augmentations and their parameters.
-   **[License Guide](../license.md):** If using AlbumentationsX, understand the dual licensing model.
