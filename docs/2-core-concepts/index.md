# Core Concepts in Albumentations

Welcome to the Core Concepts section! Understanding these fundamental building blocks is key to effectively using Albumentations to create powerful and customized augmentation pipelines for your computer vision tasks.

This section breaks down the essential mechanics of the library:

1.  **[Transforms](./transforms.md):**
    Learn about individual augmentation operations – the basic units of change like rotation, blurring, or color shifting. We'll cover what they are and how they work fundamentally.

2.  **[Pipelines](./pipelines.md):**
    Discover how to chain multiple transforms together using container transforms like `Compose`, `OneOf`, `SomeOf`, and `Sequential` to create complex augmentation workflows.

3.  **[Working with Targets](./targets.md):**
    Understand how Albumentations handles not just images, but also associated labels like segmentation masks, bounding boxes, and keypoints, ensuring transformations are applied consistently across all relevant data.

4.  **[Setting Probabilities](./probabilities.md):**
    Explore how the `p` parameter controls the probability of applying a transform or an entire pipeline, allowing you fine-grained control over the augmentation process.

By grasping these concepts, you'll be well-equipped to design augmentation strategies tailored to your specific needs. Dive into the individual pages to learn more about each component.

## Where to Go Next?

After familiarizing yourself with the core concepts, you can:

-   **[See Basic Usage Examples](../3-basic-usage):** Put the concepts into practice with code examples for common tasks like classification, segmentation, and object detection.
-   **[Learn How to Pick Augmentations](../3-basic-usage/choosing-augmentations.md):** Get guidance on selecting the right transforms for your specific problem.
-   **[Explore Advanced Guides](../4-advanced-guides):** Dive deeper into topics like creating custom transforms or serializing pipelines.
-   **[Visually Explore Transforms](https://explore.albumentations.ai):** Experiment with individual transforms and their parameters interactively.
