# Arxiv Paper: Two Levels of Image Augmentation

**Overview:** Turn the two-level augmentation framework into an arxiv preprint + JMLR submission. Path A: JMLR → TMLR → Pattern Recognition → IEEE Access. Novel contribution: in-distribution vs out-of-distribution taxonomy. One experiment (manifold distance, no training).

## Source Material Already Written

Most of the paper content exists across two docs files:

- [what-are-image-augmentations.md](../docs/1-introduction/what-are-image-augmentations.md) — two-level framework, manifold perspective, invariance/equivariance, failure modes, self-supervised/TTA/domain-randomization, practical guidelines (~640 lines)
- [choosing-augmentations.md](../docs/3-basic-usage/choosing-augmentations.md) — transform family deep-dives, task-specific design, evaluation protocol, spice rack analogy (~900 lines)

Combined these are roughly 15,000 words — more than enough raw material for a 10-12 page paper.

---

## Step 1: Literature Search and Positioning

Before writing, nail down exactly where this paper sits relative to existing surveys. Key papers to review and cite:

- **Shorten & Khoshgoftaar 2019** — "A survey on Image Data Augmentation for Deep Learning" (the most-cited survey; organizes by technique, no two-level distinction)
- **Mumuni & Mumuni 2022** — "Data augmentation: A comprehensive survey of modern approaches" (broader scope, still technique-organized)
- **Yang et al. 2023** — "Image Data Augmentation for Deep Learning: A Survey" (similar gap)
- **Xu et al. 2023** — survey with generative augmentation focus
- **Chapelle et al. 2001** — Vicinal Risk Minimization (theoretical foundation)
- **Key method papers**: AutoAugment (Cubuk 2019), RandAugment (Cubuk 2020), TrivialAugment (Muller & Hutter 2021), CutOut (DeVries 2017), MixUp (Zhang 2018), CutMix (Yun 2019), SimCLR (Chen 2020), Tobin et al. 2017 (domain randomization), Geirhos et al. 2019 (texture bias/style transfer)

**Deliverable**: A 2-page related work section that explicitly states what existing surveys do (organize by technique) and what this paper does differently (organize by purpose along the distribution axis).

## Step 2: Write the Classification Table

The core novel artifact: classify every major augmentation method as Level 1, Level 2, or mixed. This table does not exist in any published survey.

Columns: Method | Level | Mechanism | When it helps | When it hurts | Key reference

Cover at minimum: flips, rotations, crops, affine, perspective, brightness/contrast, color jitter, blur, noise, CutOut/CoarseDropout, MixUp, CutMix, Mosaic, style transfer, domain randomization, elastic transforms, weather simulation, grayscale/channel dropout, AutoAugment, RandAugment, TrivialAugment, AugMax.

## Step 3: Draft Paper Structure

```
1. Abstract
2. Introduction (the gap in existing taxonomies)
3. Background
   3.1 Augmentation as Vicinal Risk Minimization
   3.2 The Data Manifold Perspective
   3.3 Invariance vs Equivariance
4. The Two-Level Framework (core contribution)
   4.1 Level 1: In-Distribution Densification
   4.2 Level 2: Out-of-Distribution Regularization
   4.3 The Label Preservation Constraint
   4.4 Interaction: Model Capacity Determines the Mix
5. Taxonomy of Methods Through the Two-Level Lens (the table)
6. Task-Specific Design Principles
   6.1 Classification
   6.2 Object Detection
   6.3 Segmentation
   6.4 Keypoints / Pose
   6.5 Domain-Specific (Medical, Satellite, OCR, Industrial)
7. Beyond Supervised Training
   7.1 Self-Supervised / Contrastive Learning
   7.2 Test-Time Augmentation
   7.3 Domain Randomization
8. Failure Modes and Practical Guidelines
9. Open Problems
10. Conclusion
```

## Step 4: Experiment B — Manifold Distance Visualization

Single experiment, no training required. Forward passes only with one pretrained model (ResNet-50 from timm).

**Setup:**

1. Take a standard dataset (e.g., ImageNet validation subset or CIFAR-100 test)
2. Extract feature embeddings for all clean images using pretrained ResNet-50
3. For each augmentation method, apply at increasing magnitude levels
4. Extract features of augmented images
5. Compute average kNN distance (k=5 or 10) from each augmented embedding to the clean embedding set

**Expected result:** Level 1 transforms (mild rotation, brightness, flip) produce embeddings that stay close to the clean manifold across all magnitudes. Level 2 transforms (grayscale, CoarseDropout, heavy color jitter) depart from the manifold — and the departure increases with magnitude. This single figure makes the two-level distinction empirically visible.

**Scope decision:** Single pretrained model only. Multi-architecture comparison (do CNNs and ViTs define different manifolds? do Level 2 transforms depart differently?) is interesting but deferred to a separate follow-up paper.

## Step 5: Convert Docs to Paper Prose

Rewrite the existing doc content into academic style:

- Remove Albumentations-specific code examples and library links (keep 1-2 illustrative code blocks max)
- Add formal definitions where the docs use informal language
- Add citations inline where the docs just state facts
- Compress the task-specific sections (docs are exhaustive; paper needs 1-2 paragraphs each)
- Add the "open problems" section (not in docs): automatic level assignment, optimal level mixing as f(dataset size, model capacity), augmentation for foundation models, generative augmentation blurring the L1/L2 boundary

## Step 6: Figures

Key figures to create or adapt:

1. **Conceptual diagram**: Two-level framework overview (Level 1 = fill distribution gaps, Level 2 = regularization pressure, shared constraint = label preservation)
2. **Distribution diagram**: Already exists in docs — training distribution widened by augmentation, with Level 1 and Level 2 regions marked
3. **Example grid**: Same image under Level 1 transforms vs Level 2 transforms side by side
4. **The classification table** (Step 2)
5. **Failure mode examples**: Over-augmentation, label corruption
6. **Manifold distance**: Scatter plot for Experiment B

## Step 7: Write Abstract and Submit

Draft abstract (~200 words) highlighting:

- Gap: existing surveys organize augmentations by technique, missing the functional distinction
- Contribution: two-level framework (in-distribution densification vs out-of-distribution regularization)
- Key insight: "only use realistic augmentation" is incomplete; model capacity determines the optimal mix
- Scope: comprehensive survey of methods through this lens, task-specific guidelines, failure mode taxonomy

---

## Venue Strategy: Path A

```
Week 0:   arxiv preprint (establish priority)
          ↓
Attempt 1: JMLR (reach — prestigious, no page limit, accepts frameworks/surveys)
          ↓ if rejected
Attempt 2: TMLR (match — designed for conceptual contributions, fast review)
          ↓ if rejected
Attempt 3: Pattern Recognition (match — good impact, survey-friendly)
          ↓ if rejected
Attempt 4: IEEE Access (safety — peer-reviewed, fast, high acceptance)

Parallel (non-archival, no conflict): Data-Centric AI workshop at NeurIPS/ICML
```

Since JMLR is the first target, the paper should aim for their standard: comprehensive, well-cited, thorough coverage. No page limit works in our favor — we can be as detailed as needed without cutting material. Use JMLR LaTeX template from the start.

---

## Decisions Resolved

1. **Venue path**: A — JMLR first, then TMLR, then Pattern Recognition, then IEEE Access
2. **Experiments**: Experiment B only (manifold distance). No training. Single pretrained ResNet-50.
3. **Architecture scope**: Single model. Multi-architecture deferred to follow-up paper.
4. **Classification table**: 25+ methods.
5. **LaTeX template**: JMLR format.

## Future Paper Ideas (Out of Scope Here)

- **Architecture x Augmentation interaction**: How do CNNs vs ViTs vs hybrid models respond differently to Level 1 vs Level 2 augmentation? Requires training runs across model families and sizes.
- **Automatic level assignment**: Can we build a tool that classifies a transform as Level 1 or Level 2 for a given dataset automatically (e.g., using the manifold distance metric from Experiment B)?
