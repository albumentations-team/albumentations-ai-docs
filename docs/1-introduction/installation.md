# Installation

AlbumentationsX requires Python 3.10 or higher. We recommend using the latest stable Python version.

## Installation Methods

**AlbumentationsX** offers:
- ⚡ Improved performance and bug fixes
- 🔧 Active maintenance and new features
- 📊 Better support for production environments

AlbumentationsX is dual-licensed (AGPL/Commercial). For more information about licensing, see our [License Guide](../license.md).

#### Basic Installation

**Important:** Starting with AlbumentationsX 2.0.14, OpenCV is **not installed automatically**. You need to explicitly choose your OpenCV variant:

**For GUI support** (desktop environments, visualization):
```bash
pip install -U albumentationsx[gui]
```

**For headless environments** (servers, Docker, CI/CD):
```bash
pip install -U albumentationsx[headless]
```

**For OpenCV contrib modules**:
```bash
pip install -U albumentationsx[contrib]
```

**Manual OpenCV installation** (if you already have OpenCV or want full control):
```bash
pip install opencv-python  # or opencv-python-headless, opencv-contrib-python, etc.
pip install -U albumentationsx
```

Here's a basic example:
```python
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

## Understanding OpenCV Dependencies

AlbumentationsX relies heavily on OpenCV for image processing operations.

### Why OpenCV is now optional (AlbumentationsX 2.0.14+)

Previously, AlbumentationsX tried to automatically manage OpenCV installation, which caused persistent issues:
- Conflicting OpenCV packages (`opencv-python` and `opencv-python-headless`) could be installed simultaneously
- GUI features like `cv2.imshow()` would break unexpectedly
- Import order became unpredictable

**The root cause:** Python's build system evaluates dependencies in isolated environments, making it impossible to reliably detect what's already installed in your environment.

### The new approach

Starting with version 2.0.14, AlbumentationsX:
- ✅ Does **not** install OpenCV automatically
- ✅ Does **not** try to guess which variant you need
- ✅ Will **never** install conflicting OpenCV packages
- ✅ Gives you explicit control over your environment

### Choosing the right OpenCV variant

**`albumentationsx[gui]`** - Installs `opencv-python`:
- Use for desktop applications
- Includes GUI support (`cv2.imshow()`, `cv2.waitKey()`, etc.)
- Larger installation size

**`albumentationsx[headless]`** - Installs `opencv-python-headless`:
- Use for servers, Docker containers, CI/CD pipelines
- No GUI dependencies
- Smaller installation size

**`albumentationsx[contrib]`** - Installs `opencv-contrib-python`:
- Includes additional OpenCV modules
- Use when you need extended functionality

**Manual installation** - Install OpenCV separately:
- Maximum control over the exact version and variant
- Useful if you have custom OpenCV builds
- Install your preferred OpenCV package first, then AlbumentationsX

### Why this is better

This change trades a bit of automation for:
- 🔒 Stable, predictable environments
- 🎯 Clear, explicit behavior
- 🚫 No silent breakage or conflicts
- 💡 Fewer "why did this suddenly stop working?" moments

**Less magic, more reliability.**

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
