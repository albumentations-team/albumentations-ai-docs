# Installation

Albumentations requires Python 3.9 or higher. We recommend using the latest stable Python version.

## Installation Methods

### From PyPI (Recommended Stable Version)

This installs the latest official release.

```bash
pip install -U albumentations
```

### From Conda Forge

If you are using Anaconda or Miniconda:

```bash
conda install -c conda-forge albumentations
```

### From GitHub (Latest Development Version)

This installs the bleeding-edge version directly from the `main` branch.

```bash
pip install -U git+https://github.com/albumentations-team/albumentations
```

**Note:** Installing from the `main` branch might give you newer features but could potentially be less stable than official releases.

## Handling OpenCV Dependencies

Albumentations relies heavily on OpenCV.

*   **Default:** By default, Albumentations depends on `opencv-python-headless`. This version is chosen because it avoids installing GUI-related dependencies, making it suitable for server environments and containers where graphical interfaces are not needed.
*   **Using Your Existing OpenCV:** If you already have a different OpenCV distribution installed (like `opencv-python`, `opencv-contrib-python`, or `opencv-contrib-python-headless`), pip should automatically detect and use it.
*   **Forcing Source Build (Advanced):** If you need to force pip to build Albumentations from source and use a *specific*, pre-existing OpenCV installation (perhaps compiled manually), you can use the `--no-binary albumentations` flag:
    ```bash
    pip install -U albumentations --no-binary albumentations
    ```
    In most standard cases, this flag is **not** required.

## Verify Installation

After installation, you can verify it by running:

```bash
python -c "import albumentations as A; print(A.__version__)"
```

This should print the installed version number of Albumentations.

## Where to Go Next?

Now that you have Albumentations installed, here are some logical next steps:

-   **[Understand Core Concepts](../2-core-concepts/index.md):** Learn about transforms, pipelines, targets, and probabilities - the fundamental building blocks of Albumentations.
-   **[See Basic Usage Examples](../3-basic-usage/index.md):** Explore how to apply augmentations for common computer vision tasks.
-   **[Explore Transforms](https://explore.albumentations.ai):** Visually experiment with different augmentations and their parameters.
