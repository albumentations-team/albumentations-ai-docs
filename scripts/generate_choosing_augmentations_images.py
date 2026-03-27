"""Generate all images for docs/3-basic-usage/choosing-augmentations.md

Requires: pip install albumentationsx
Run:      python3.12 scripts/generate_choosing_augmentations_images.py

Input assets (in temp_img/):
    fish.webp           — colorful fish (640x703)
    soccer.web          — soccer field (2690x1793)
    soccer.json         — LabelMe annotation (ball bbox, person bboxes + polygon masks)
    person_small.webp   — person far from camera
    person_medium.webp  — person at medium distance
    person_large.webp   — person close to camera

Output (all .webp in img/basic_usage/choosing_augmentations/):
    header.webp
    level1_vs_level2.webp
    constrained_vs_unconstrained_dropout.webp
    train_hard_test_easy.webp
    pipeline_steps.webp
    augmentation_bug_bbox.webp
    scale_variation.webp
    digit_rotation_6_vs_9.webp
    domain_transforms_sampler.webp
    diagnostic_results_table.webp
    pipeline_output_examples.webp
    crop_resize_letterbox.webp
    aggregate_hides_regression.webp
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMP_IMG = ROOT / "temp_img"
FISH_PATH = TEMP_IMG / "fish.webp"
SOCCER_IMAGE_PATH = TEMP_IMG / "soccer.web"
SOCCER_JSON_PATH = TEMP_IMG / "soccer.json"
PERSON_SMALL_PATH = TEMP_IMG / "person_small.webp"
PERSON_MEDIUM_PATH = TEMP_IMG / "person_medium.webp"
PERSON_LARGE_PATH = TEMP_IMG / "person_large.webp"
OUT_DIR = ROOT / "img" / "basic_usage" / "choosing_augmentations"

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD = 10
BAR_H = 30
TALL_BAR_H = 44

BG = 245


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_webp(path: Path, img: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")
    print(f"  Saved {path.relative_to(ROOT)}  ({img.shape[1]}x{img.shape[0]})")


def resize_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def resize_to_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0:y0 + side, x0:x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def resize_to_fit(img: np.ndarray, max_h: int, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_h / h, max_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def add_label_bar(img: np.ndarray, text: str, bg: tuple = (0, 0, 0),
                  fg: tuple = (255, 255, 255), bar_h: int = BAR_H) -> np.ndarray:
    w = img.shape[1]
    bar = np.full((bar_h, w, 3), bg, dtype=np.uint8)
    font_scale = 0.45 if bar_h > BAR_H else 0.5
    cv2.putText(bar, text, (6, bar_h - 8), FONT, font_scale, fg, 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def add_two_line_label(img: np.ndarray, line1: str, line2: str,
                       bg: tuple = (0, 0, 0), fg: tuple = (255, 255, 255)) -> np.ndarray:
    w = img.shape[1]
    bar = np.full((TALL_BAR_H, w, 3), bg, dtype=np.uint8)
    cv2.putText(bar, line1, (6, 16), FONT, 0.42, fg, 1, cv2.LINE_AA)
    cv2.putText(bar, line2, (6, 36), FONT, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def hstack_with_gap(images: list[np.ndarray], gap: int = PAD,
                    bg_val: int = BG) -> np.ndarray:
    max_h = max(img.shape[0] for img in images)
    parts = []
    for i, img in enumerate(images):
        if img.shape[0] < max_h:
            fill = np.full((max_h - img.shape[0], img.shape[1], 3), bg_val, dtype=np.uint8)
            img = np.concatenate([img, fill], axis=0)
        parts.append(img)
        if i < len(images) - 1:
            parts.append(np.full((max_h, gap, 3), bg_val, dtype=np.uint8))
    return np.concatenate(parts, axis=1)


def vstack_with_gap(images: list[np.ndarray], gap: int = PAD,
                    bg_val: int = BG) -> np.ndarray:
    max_w = max(img.shape[1] for img in images)
    parts = []
    for i, img in enumerate(images):
        if img.shape[1] < max_w:
            fill = np.full((img.shape[0], max_w - img.shape[1], 3), bg_val, dtype=np.uint8)
            img = np.concatenate([img, fill], axis=1)
        parts.append(img)
        if i < len(images) - 1:
            parts.append(np.full((gap, max_w, 3), bg_val, dtype=np.uint8))
    return np.concatenate(parts, axis=0)


def section_header(width: int, text: str, bg: tuple = (50, 50, 50),
                   fg: tuple = (255, 255, 255)) -> np.ndarray:
    bar = np.full((34, width, 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (10, 24), FONT, 0.6, fg, 2, cv2.LINE_AA)
    return bar


def draw_bboxes_on(img: np.ndarray, bboxes: list, labels: list,
                   colors: dict | None = None, thickness: int = 3) -> np.ndarray:
    default_colors = {"ball": (0, 255, 255), "person": (0, 255, 0)}
    if colors is None:
        colors = default_colors
    out = img.copy()
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, label, (x1 + 2, y1 + 16), FONT, 0.45, color, 1, cv2.LINE_AA)
    return out


def overlay_mask(img: np.ndarray, mask: np.ndarray,
                 color: tuple = (0, 200, 100), alpha: float = 0.35) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def read_soccer_annotations(json_path: Path) -> tuple:
    with open(json_path) as f:
        ann = json.load(f)
    bboxes, labels, polygons = [], [], []
    for shape in ann["shapes"]:
        if shape["shape_type"] == "rectangle":
            pts = shape["points"]
            bboxes.append((pts[0][0], pts[0][1], pts[1][0], pts[1][1]))
            labels.append(shape["label"])
        elif shape["shape_type"] == "polygon":
            polygons.append(np.array(shape["points"], dtype=np.float32))
    return bboxes, labels, polygons


def create_mask_from_polygons(polygons: list, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for pts in polygons:
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    return mask


def get_applied_names(result: dict) -> str:
    if "applied_transforms" not in result:
        return ""
    names = [name for name, _ in result["applied_transforms"]]
    short = {"HorizontalFlip": "HFlip", "RandomBrightnessContrast": "BrCon",
             "PhotoMetricDistort": "PhotoDist", "CoarseDropout": "Dropout",
             "GridDropout": "GridDrop", "GaussianBlur": "GBlur",
             "MotionBlur": "MBlur", "GaussNoise": "Noise",
             "SmallestMaxSize": "Resize", "LongestMaxSize": "Resize",
             "RandomCrop": "RCrop", "CenterCrop": "CCrop",
             "PadIfNeeded": "Pad", "Normalize": "Norm",
             "RandomResizedCrop": "RRCrop", "SquareSymmetry": "SqSym",
             "Affine": "Affine", "ToGray": "Gray",
             "ChannelDropout": "ChDrop"}
    abbreviated = [short.get(n, n) for n in names
                   if n not in ("SmallestMaxSize", "LongestMaxSize", "PadIfNeeded",
                                "RandomCrop", "CenterCrop", "Normalize",
                                "RandomResizedCrop")]
    return ", ".join(abbreviated) if abbreviated else "(no stochastic)"


# ---------------------------------------------------------------------------
# Image 0: Header mosaic
# ---------------------------------------------------------------------------
def build_header(fish: np.ndarray) -> None:
    print("Building header...")
    cell_size = 160
    cols, rows = 8, 4
    base = resize_to_square(fish, cell_size)

    transforms = [
        A.NoOp(), A.HorizontalFlip(p=1.0),
        A.Rotate(limit=(20, 20), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Rotate(limit=(-35, -35), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Affine(scale=0.6, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Affine(shear=25, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Perspective(scale=0.15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.ElasticTransform(alpha=80, sigma=6, p=1.0),
        A.PhotoMetricDistort(brightness_range=(1.4, 1.7), contrast_range=(1.5, 2.0),
                             saturation_range=(1.8, 2.5), hue_range=(-0.15, 0.15), p=1.0),
        A.PhotoMetricDistort(brightness_range=(0.3, 0.55), contrast_range=(0.2, 0.5),
                             saturation_range=(0.1, 0.4), hue_range=(-0.05, 0.05), p=1.0),
        A.PhotoMetricDistort(brightness_range=(0.7, 1.3), contrast_range=(0.4, 2.0),
                             saturation_range=(0.2, 2.5), hue_range=(-0.2, 0.2), p=1.0),
        A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=0, p=1.0),
        A.ToGray(p=1.0), A.Posterize(num_bits=2, p=1.0),
        A.Equalize(p=1.0), A.Solarize(threshold_range=(0.45, 0.55), p=1.0),
        A.GaussianBlur(blur_limit=(13, 13), p=1.0),
        A.MotionBlur(blur_limit=(17, 17), p=1.0),
        A.GaussNoise(std_range=(0.12, 0.14), p=1.0),
        A.ISONoise(color_shift=(0.1, 0.3), intensity=(0.7, 0.9), p=1.0),
        A.Defocus(radius=(5, 5), alias_blur=0.1, p=1.0),
        A.ImageCompression(quality_range=(5, 8), p=1.0),
        A.Downscale(scale_range=(0.2, 0.2), p=1.0),
        A.Sharpen(alpha=(0.9, 0.9), lightness=(1.0, 1.0), p=1.0),
        A.CoarseDropout(num_holes_range=(6, 8), hole_height_range=(0.07, 0.12),
                        hole_width_range=(0.07, 0.12), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(6, 8), hole_height_range=(0.07, 0.12),
                        hole_width_range=(0.07, 0.12), fill="random", p=1.0),
        A.GridDropout(ratio=0.45, p=1.0),
        A.ChannelDropout(channel_drop_range=(1, 1), p=1.0),
        A.RandomFog(fog_coef_range=(0.5, 0.7), p=1.0),
        A.RandomRain(slant_range=(-10, 10), drop_length=15, drop_width=1, p=1.0),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1.0),
        A.PlanckianJitter(mode="blackbody", p=1.0),
    ]

    cells = [A.Compose([t])(image=base)["image"] for t in transforms[:cols * rows]]
    row_imgs = [hstack_with_gap(cells[r * cols:(r + 1) * cols], gap=2, bg_val=20)
                for r in range(rows)]
    save_webp(OUT_DIR / "header.webp", vstack_with_gap(row_imgs, gap=2, bg_val=20))


# ---------------------------------------------------------------------------
# Image 1: Level 1 vs Level 2
# ---------------------------------------------------------------------------
def build_level1_vs_level2(fish: np.ndarray) -> None:
    print("Building level1_vs_level2...")
    cell_h = 260
    base = resize_height(fish, cell_h)

    level1_ops = [
        ("Original", A.NoOp()),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("Rotate +8°", A.Rotate(limit=(8, 8), border_mode=cv2.BORDER_CONSTANT, p=1.0)),
        ("Brightness +15%", A.RandomBrightnessContrast(
            brightness_limit=(0.15, 0.15), contrast_limit=0, p=1.0)),
        ("Slight blur", A.GaussianBlur(blur_limit=(5, 5), p=1.0)),
    ]

    level2_ops = [
        ("Original", A.NoOp()),
        ("ToGray", A.ToGray(p=1.0)),
        ("CoarseDropout", A.CoarseDropout(
            num_holes_range=(6, 6), hole_height_range=(0.08, 0.15),
            hole_width_range=(0.08, 0.15), fill=0, p=1.0)),
        ("PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
        ("ChannelDropout", A.ChannelDropout(channel_drop_range=(1, 1), p=1.0)),
    ]

    l1_cells = [add_label_bar(A.Compose([op])(image=base)["image"], n)
                for n, op in level1_ops]
    l2_cells = [add_label_bar(A.Compose([op])(image=base)["image"], n)
                for n, op in level2_ops]

    row1 = hstack_with_gap(l1_cells)
    row2 = hstack_with_gap(l2_cells)
    h1 = section_header(row1.shape[1],
                        "Level 1: In-Distribution (densification)", bg=(40, 100, 50))
    h2 = section_header(row2.shape[1],
                        "Level 2: Out-of-Distribution (regularization)", bg=(100, 40, 50))
    save_webp(OUT_DIR / "level1_vs_level2.webp",
              vstack_with_gap([h1, row1, h2, row2], gap=6))


# ---------------------------------------------------------------------------
# Image 2: Constrained vs Unconstrained Dropout (soccer ball)
# ---------------------------------------------------------------------------
def build_constrained_dropout(soccer: np.ndarray, bboxes: list,
                              labels: list) -> None:
    print("Building constrained_vs_unconstrained_dropout...")
    cell_h = 450
    ball_idx = labels.index("ball")
    ball_bbox = bboxes[ball_idx]

    t_resize = A.Compose([
        A.LongestMaxSize(max_size=int(cell_h * soccer.shape[1] / soccer.shape[0]), p=1.0),
        A.SmallestMaxSize(max_size=cell_h, p=1.0),
    ], bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]))

    resized = t_resize(image=soccer, bboxes=[ball_bbox], labels=["ball"])
    img_r, bbox_r = resized["image"], resized["bboxes"][0]

    np.random.seed(42)
    random.seed(42)
    unc = A.Compose([
        A.CoarseDropout(num_holes_range=(20, 20), hole_height_range=(0.04, 0.08),
                        hole_width_range=(0.04, 0.08), fill=(255, 0, 0), p=1.0),
    ], bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]))
    unc_r = unc(image=img_r, bboxes=[bbox_r], labels=["ball"])
    unc_img = draw_bboxes_on(unc_r["image"], unc_r["bboxes"], ["ball"],
                             colors={"ball": (0, 255, 0)})

    np.random.seed(42)
    random.seed(42)
    con = A.Compose([
        A.ConstrainedCoarseDropout(num_holes_range=(1, 1), hole_height_range=(0.5, 0.5),
                                   hole_width_range=(0.5, 0.5), fill=(255, 0, 0),
                                   bbox_labels=["ball"], p=1.0),
    ], bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]))
    con_r = con(image=img_r, bboxes=[bbox_r], labels=["ball"])
    con_img = draw_bboxes_on(con_r["image"], con_r["bboxes"], ["ball"],
                             colors={"ball": (0, 255, 0)})

    orig_img = draw_bboxes_on(img_r, [bbox_r], ["ball"], colors={"ball": (0, 255, 0)})
    cells = [
        add_label_bar(orig_img, "Original (ball bbox)"),
        add_label_bar(unc_img, "CoarseDropout (random, misses ball)"),
        add_label_bar(con_img, "ConstrainedCoarseDropout (targets ball)"),
    ]
    save_webp(OUT_DIR / "constrained_vs_unconstrained_dropout.webp",
              hstack_with_gap(cells))


# ---------------------------------------------------------------------------
# Image 3: Train Hard, Test Easy
# ---------------------------------------------------------------------------
def build_train_hard_test_easy(fish: np.ndarray) -> None:
    print("Building train_hard_test_easy...")
    cell_h = 200
    base = resize_height(fish, cell_h)

    heavy_ops = [
        ("ToGray", A.ToGray(p=1.0)),
        ("CoarseDropout", A.CoarseDropout(
            num_holes_range=(8, 8), hole_height_range=(0.08, 0.16),
            hole_width_range=(0.08, 0.16), fill=0, p=1.0)),
        ("PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
        ("Blur + noise", A.Compose([
            A.GaussianBlur(blur_limit=(9, 9), p=1.0),
            A.GaussNoise(std_range=(0.08, 0.08), p=1.0),
        ])),
    ]

    train_cells = []
    for name, op in heavy_ops:
        img = (op(image=base)["image"] if isinstance(op, A.Compose)
               else A.Compose([op])(image=base)["image"])
        train_cells.append(add_label_bar(img, name))

    train_block = hstack_with_gap(train_cells, gap=6)
    total_h = train_block.shape[0]

    # Arrow
    arrow_w = 70
    arrow = np.full((total_h, arrow_w, 3), BG, dtype=np.uint8)
    mid_y = total_h // 2
    cv2.rectangle(arrow, (12, mid_y - 4), (arrow_w - 22, mid_y + 4), (80, 80, 80), -1)
    tip = np.array([[arrow_w - 8, mid_y], [arrow_w - 26, mid_y - 18],
                    [arrow_w - 26, mid_y + 18]], dtype=np.int32)
    cv2.fillPoly(arrow, [tip], (80, 80, 80))

    clean_cell = add_label_bar(base, "Clean image", bg=(40, 100, 60))
    if clean_cell.shape[0] < total_h:
        pad_b = np.full((total_h - clean_cell.shape[0], clean_cell.shape[1], 3),
                        BG, dtype=np.uint8)
        clean_cell = np.concatenate([clean_cell, pad_b], axis=0)

    strip = np.concatenate([train_block, arrow, clean_cell], axis=1)
    total_w = strip.shape[1]
    train_w = train_block.shape[1]
    infer_start = train_w + arrow_w

    header = np.full((32, total_w, 3), BG, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (train_w, 32), (140, 50, 50), -1)
    cv2.putText(header, "Training: hard (augmented)", (8, 22),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(header, (infer_start, 0), (total_w, 32), (40, 100, 60), -1)
    cv2.putText(header, "Inference: easy", (infer_start + 8, 22),
                FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    save_webp(OUT_DIR / "train_hard_test_easy.webp",
              np.concatenate([header, strip], axis=0))


# ---------------------------------------------------------------------------
# Image 4: Pipeline Steps — 6 visible steps (Normalize not visualizable)
# ---------------------------------------------------------------------------
def build_pipeline_steps(fish: np.ndarray) -> None:
    print("Building pipeline_steps...")
    cell_h = 250
    base = resize_height(fish, cell_h)

    steps = [
        ("Step 1: Crop", A.RandomResizedCrop(
            size=(cell_h, base.shape[1]), scale=(0.5, 0.7), ratio=(0.7, 0.9), p=1.0)),
        ("Step 2: HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("Step 3: CoarseDropout", A.CoarseDropout(
            num_holes_range=(6, 6), hole_height_range=(0.08, 0.15),
            hole_width_range=(0.08, 0.15), fill=0, p=1.0)),
        ("Step 4: ToGray", A.ToGray(p=1.0)),
        ("Step 5: Affine", A.Affine(
            scale=(0.85, 0.85), rotate=(12, 12),
            border_mode=cv2.BORDER_CONSTANT, p=1.0)),
        ("Step 6: PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
    ]

    cells = [add_label_bar(base, "Original")]
    for name, op in steps:
        img = A.Compose([op])(image=base.copy())["image"]
        cells.append(add_label_bar(img, name))

    # 7 cells -> 4+3 with an info cell for Normalize
    norm_cell = np.full((cell_h + BAR_H, cells[0].shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(norm_cell, "Step 7: Normalize", (10, cell_h // 2 - 10),
                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(norm_cell, "(scales values to", (10, cell_h // 2 + 15),
                FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(norm_cell, "model's expected range.", (10, cell_h // 2 + 35),
                FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(norm_cell, "Always last.)", (10, cell_h // 2 + 55),
                FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cells.append(norm_cell)

    row1 = hstack_with_gap(cells[:4])
    row2 = hstack_with_gap(cells[4:])
    save_webp(OUT_DIR / "pipeline_steps.webp", vstack_with_gap([row1, row2]))


# ---------------------------------------------------------------------------
# Image 5: Augmentation Bug — bbox not following flip
# ---------------------------------------------------------------------------
def build_augmentation_bug(soccer: np.ndarray, bboxes: list,
                           labels: list) -> None:
    print("Building augmentation_bug_bbox...")
    cell_h = 400
    ball_idx = labels.index("ball")
    ball_bbox = bboxes[ball_idx]

    t_resize = A.Compose([
        A.SmallestMaxSize(max_size=cell_h, p=1.0),
    ], bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]))
    resized = t_resize(image=soccer, bboxes=[ball_bbox], labels=["ball"])
    base, bbox = resized["image"], resized["bboxes"][0]

    original = draw_bboxes_on(base, [bbox], ["ball"], colors={"ball": (0, 255, 0)})

    correct = A.Compose(
        [A.HorizontalFlip(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["labels"]),
    )(image=base, bboxes=[bbox], labels=["ball"])
    correct_img = draw_bboxes_on(correct["image"], correct["bboxes"], ["ball"],
                                 colors={"ball": (0, 255, 0)})

    bug_img = draw_bboxes_on(cv2.flip(base, 1), [bbox], ["ball"],
                             colors={"ball": (255, 0, 0)})

    cells = [
        add_label_bar(original, "Original"),
        add_label_bar(bug_img, "BUG: image flipped, bbox not updated"),
        add_label_bar(correct_img, "Correct: bbox follows the flip"),
    ]
    save_webp(OUT_DIR / "augmentation_bug_bbox.webp", hstack_with_gap(cells))


# ---------------------------------------------------------------------------
# Image 6: Scale Variation
# ---------------------------------------------------------------------------
def build_scale_variation(person_small: np.ndarray, person_medium: np.ndarray,
                          person_large: np.ndarray) -> None:
    print("Building scale_variation...")
    cell_h, cell_w = 350, 280
    cells = []
    for img, label in [
        (person_small, "Far (~50m): person is ~30px tall"),
        (person_medium, "Medium (~10m): person is ~150px tall"),
        (person_large, "Close (~2m): person fills the frame"),
    ]:
        resized = resize_to_fit(img, cell_h, cell_w)
        h, w = resized.shape[:2]
        canvas = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
        y0, x0 = (cell_h - h) // 2, (cell_w - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = resized
        cells.append(add_label_bar(canvas, label, bg=(40, 40, 40)))
    save_webp(OUT_DIR / "scale_variation.webp",
              hstack_with_gap(cells, gap=6, bg_val=30))


# ---------------------------------------------------------------------------
# Image 7: Digit 6 vs 9 rotation
# ---------------------------------------------------------------------------
def build_digit_rotation() -> None:
    print("Building digit_rotation_6_vs_9...")
    size = 200

    def draw_digit(d: str) -> np.ndarray:
        canvas = np.full((size, size, 3), 240, dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(d, FONT, 5.0, 12)
        cv2.putText(canvas, d, ((size - tw) // 2, (size + th) // 2),
                    FONT, 5.0, (30, 30, 30), 12, cv2.LINE_AA)
        return canvas

    d6, d9 = draw_digit("6"), draw_digit("9")
    rot_90 = A.Compose([A.Rotate(limit=(90, 90), border_mode=cv2.BORDER_CONSTANT, p=1.0)])
    rot_180 = A.Compose([A.Rotate(limit=(180, 180), border_mode=cv2.BORDER_CONSTANT, p=1.0)])

    row1 = hstack_with_gap([
        add_label_bar(d6, "6 (original)", bg=(40, 100, 50)),
        add_label_bar(rot_90(image=d6)["image"], "6 rotated 90°", bg=(40, 100, 50)),
        add_label_bar(rot_180(image=d6)["image"], "6 rotated 180° = 9!", bg=(180, 40, 40)),
    ])
    row2 = hstack_with_gap([
        add_label_bar(d9, "9 (original)", bg=(40, 100, 50)),
        add_label_bar(rot_180(image=d9)["image"], "9 rotated 180° = 6!", bg=(180, 40, 40)),
    ])

    h1 = section_header(row1.shape[1], "Rotating digit 6: 180° corrupts label", bg=(100, 40, 40))
    h2 = section_header(row2.shape[1], "Rotating digit 9: same problem", bg=(100, 40, 40))
    save_webp(OUT_DIR / "digit_rotation_6_vs_9.webp",
              vstack_with_gap([h1, row1, h2, row2], gap=8))


# ---------------------------------------------------------------------------
# Image 8: Domain-Specific Transform Sampler
# ---------------------------------------------------------------------------
def build_domain_transforms_sampler(fish: np.ndarray) -> None:
    print("Building domain_transforms_sampler...")
    cell_h = 180
    base = resize_height(fish, cell_h)

    categories = [
        ("Color & Lighting", [
            ("PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
            ("PlanckianJitter", A.PlanckianJitter(mode="blackbody", p=1.0)),
            ("RandomToneCurve", A.RandomToneCurve(scale=0.3, p=1.0)),
            ("CLAHE", A.CLAHE(clip_limit=6.0, p=1.0)),
        ], (60, 120, 60)),
        ("Blur & Noise", [
            ("GaussianBlur", A.GaussianBlur(blur_limit=(11, 11), p=1.0)),
            ("MotionBlur", A.MotionBlur(blur_limit=(15, 15), p=1.0)),
            ("GaussNoise", A.GaussNoise(std_range=(0.1, 0.1), p=1.0)),
            ("ISONoise", A.ISONoise(color_shift=(0.1, 0.3), intensity=(0.7, 0.7), p=1.0)),
        ], (60, 60, 120)),
        ("Weather & Environment", [
            ("RandomFog", A.RandomFog(fog_coef_range=(0.5, 0.7), p=1.0)),
            ("RandomRain", A.RandomRain(slant_range=(-10, 10), drop_length=15, p=1.0)),
            ("RandomShadow", A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1.0)),
            ("RandomSnow", A.RandomSnow(snow_point_range=(0.3, 0.5), p=1.0)),
        ], (120, 60, 60)),
        ("Compression & Quality", [
            ("JPEG q=5", A.ImageCompression(quality_range=(5, 8), p=1.0)),
            ("Downscale 0.25x", A.Downscale(scale_range=(0.25, 0.25), p=1.0)),
            ("Defocus", A.Defocus(radius=(5, 5), alias_blur=0.1, p=1.0)),
            ("SaltAndPepper", A.SaltAndPepper(amount=(0.05, 0.05), p=1.0)),
        ], (80, 80, 80)),
    ]

    rows = []
    for cat_name, ops, header_bg in categories:
        cells = [add_label_bar(A.Compose([op])(image=base.copy())["image"], name)
                 for name, op in ops]
        row = hstack_with_gap(cells, gap=4)
        rows.extend([section_header(row.shape[1], cat_name, bg=header_bg), row])
    save_webp(OUT_DIR / "domain_transforms_sampler.webp",
              vstack_with_gap(rows, gap=4))


# ---------------------------------------------------------------------------
# Image 9: Diagnostic Results Table
# ---------------------------------------------------------------------------
def build_diagnostic_results_table() -> None:
    print("Building diagnostic_results_table...")
    W, ROW_H, HEADER_H = 820, 34, 38
    col_widths = [200, 100, 100, 100, 200]
    headers = ["Robustness Test", "Accuracy", "Delta", "Status", "Action"]
    rows_data = [
        ("Clean baseline", "94.2%", "—", "baseline", ""),
        ("HorizontalFlip", "93.8%", "-0.4%", "ok", "No action needed"),
        ("Brightness -30%", "78.1%", "-16.1%", "fail", "Add RandomBrightnessContrast"),
        ("MotionBlur k=7", "89.4%", "-4.8%", "warn", "Add MotionBlur p=0.1"),
        ("GaussNoise 0.08", "91.7%", "-2.5%", "ok", "Monitor"),
        ("Rotate ±15°", "92.1%", "-2.1%", "ok", "Already in pipeline"),
        ("Fog coef=0.5", "71.3%", "-22.9%", "fail", "Add RandomFog p=0.15"),
        ("JPEG q=10", "90.8%", "-3.4%", "warn", "Add ImageCompression"),
    ]
    total_h = HEADER_H + ROW_H * len(rows_data) + 2
    canvas = np.full((total_h, W, 3), 255, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (W, HEADER_H), (50, 50, 70), -1)
    x = 0
    for hdr, w in zip(headers, col_widths):
        cv2.putText(canvas, hdr, (x + 8, HEADER_H - 12), FONT, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += w

    status_colors = {"baseline": (220, 220, 220), "ok": (200, 240, 200),
                     "warn": (255, 240, 200), "fail": (255, 210, 210)}
    for r, (test, acc, delta, status, action) in enumerate(rows_data):
        y0 = HEADER_H + r * ROW_H
        y1 = y0 + ROW_H
        cv2.rectangle(canvas, (0, y0), (W, y1), status_colors[status], -1)
        cv2.line(canvas, (0, y1), (W, y1), (200, 200, 200), 1)
        text_color = (180, 40, 40) if status == "fail" else (30, 30, 30)
        x = 0
        for val, w in zip([test, acc, delta, status.upper(), action], col_widths):
            cv2.putText(canvas, val, (x + 8, y1 - 10), FONT, 0.40,
                        text_color, 1, cv2.LINE_AA)
            x += w

    x = 0
    for w in col_widths[:-1]:
        x += w
        cv2.line(canvas, (x, 0), (x, total_h), (180, 180, 180), 1)
    save_webp(OUT_DIR / "diagnostic_results_table.webp", canvas)


# ---------------------------------------------------------------------------
# Image 10: Pipeline Output Examples — soccer with bboxes/masks per task
#
# Strategy: each variant is a deterministic sub-pipeline with p=1 for exactly
# the transforms we want to show. This guarantees visible, diverse combinations
# without relying on random seeds to "land" on interesting combinations.
# ---------------------------------------------------------------------------
def build_pipeline_output_examples(soccer: np.ndarray, bboxes: list,
                                   labels: list, polygons: list) -> None:
    print("Building pipeline_output_examples...")
    cell_h = 360
    h_orig, w_orig = soccer.shape[:2]
    mask_full = create_mask_from_polygons(polygons, h_orig, w_orig)

    bbox_p = A.BboxParams(coord_format="pascal_voc", min_visibility=0.3,
                          label_fields=["labels"])

    # Shared resize step applied first to all pipelines so crop is sane
    def resize_cls(img):
        return A.Compose([
            A.SmallestMaxSize(max_size_hw=(cell_h, cell_h), p=1.0),
            A.RandomCrop(height=cell_h, width=cell_h, p=1.0),
        ])(image=img)["image"]

    def resize_det(img, bb, lb):
        r = A.Compose([
            A.LongestMaxSize(max_size=cell_h, p=1.0),
            A.PadIfNeeded(min_height=cell_h, min_width=cell_h,
                          border_mode=cv2.BORDER_CONSTANT, p=1.0),
        ], bbox_params=bbox_p)(image=img, bboxes=bb, labels=lb)
        return r["image"], r["bboxes"], r["labels"]

    def resize_seg(img, msk):
        r = A.Compose([
            A.SmallestMaxSize(max_size_hw=(cell_h, cell_h), p=1.0),
            A.RandomCrop(height=cell_h, width=cell_h, p=1.0),
        ])(image=img, mask=msk)
        return r["image"], r["mask"]

    # ------------------------------------------------------------------
    # Classification: 4 deterministic variants
    # ------------------------------------------------------------------
    cls_base = resize_cls(soccer.copy())

    cls_variants = [
        ("HFlip", A.Compose([
            A.HorizontalFlip(p=1.0),
        ], save_applied_params=True)),
        ("HFlip + Affine", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(1.15, 1.15), rotate=(12, 12), balanced_scale=True, p=1.0),
        ], save_applied_params=True)),
        ("Dropout + PhotoDist", A.Compose([
            A.CoarseDropout(num_holes_range=(5, 5), hole_height_range=(0.08, 0.08),
                            hole_width_range=(0.08, 0.08), p=1.0),
            A.PhotoMetricDistort(brightness_range=(0.6, 1.5), contrast_range=(0.3, 2.2),
                                 saturation_range=(0.1, 2.8), hue_range=(-0.2, 0.2), p=1.0),
        ], save_applied_params=True)),
        ("HFlip + Affine + Dropout + PhotoDist", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(0.85, 0.85), rotate=(-10, -10), balanced_scale=True, p=1.0),
            A.CoarseDropout(num_holes_range=(4, 4), hole_height_range=(0.1, 0.1),
                            hole_width_range=(0.1, 0.1), p=1.0),
            A.PhotoMetricDistort(brightness_range=(0.5, 1.6), contrast_range=(0.3, 2.5),
                                 saturation_range=(0.1, 3.0), hue_range=(-0.25, 0.25), p=1.0),
        ], save_applied_params=True)),
    ]

    cls_cells = []
    for i, (label, pipe) in enumerate(cls_variants):
        result = pipe(image=cls_base.copy())
        applied = get_applied_names(result)
        cls_cells.append(add_two_line_label(result["image"], f"Variant {i+1}", applied))

    # ------------------------------------------------------------------
    # Detection: 4 deterministic variants (no rotation — HBB boxes)
    # ------------------------------------------------------------------
    det_base_img, det_base_bboxes, det_base_labels = resize_det(
        soccer.copy(), list(bboxes), list(labels))

    det_variants = [
        ("HFlip", A.Compose([
            A.HorizontalFlip(p=1.0),
        ], bbox_params=bbox_p, save_applied_params=True)),
        ("HFlip + Scale zoom-out", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(0.65, 0.65), balanced_scale=True, p=1.0),
        ], bbox_params=bbox_p, save_applied_params=True)),
        ("Scale zoom-in + Dropout", A.Compose([
            A.Affine(scale=(1.35, 1.35), balanced_scale=True, p=1.0),
            A.CoarseDropout(num_holes_range=(4, 4), hole_height_range=(0.07, 0.07),
                            hole_width_range=(0.07, 0.07), p=1.0),
        ], bbox_params=bbox_p, save_applied_params=True)),
        ("HFlip + MotionBlur + Scale", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(0.75, 0.75), balanced_scale=True, p=1.0),
            A.MotionBlur(blur_limit=(9, 9), p=1.0),
        ], bbox_params=bbox_p, save_applied_params=True)),
    ]

    det_cells = []
    for i, (label, pipe) in enumerate(det_variants):
        result = pipe(image=det_base_img.copy(),
                      bboxes=list(det_base_bboxes), labels=list(det_base_labels))
        applied = get_applied_names(result)
        vis = draw_bboxes_on(result["image"], result["bboxes"], result["labels"])
        det_cells.append(add_two_line_label(vis, f"Variant {i+1}", applied))

    # ------------------------------------------------------------------
    # Segmentation: 4 deterministic variants (image + mask overlay)
    # ------------------------------------------------------------------
    seg_base_img, seg_base_mask = resize_seg(soccer.copy(), mask_full.copy())

    seg_variants = [
        ("HFlip", A.Compose([
            A.HorizontalFlip(p=1.0),
        ], save_applied_params=True)),
        ("HFlip + Affine", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(1.2, 1.2), rotate=(8, 8), balanced_scale=True,
                     mask_interpolation=cv2.INTER_NEAREST, p=1.0),
        ], save_applied_params=True)),
        ("Affine + Dropout", A.Compose([
            A.Affine(scale=(0.8, 0.8), rotate=(-10, -10), balanced_scale=True,
                     mask_interpolation=cv2.INTER_NEAREST, p=1.0),
            A.CoarseDropout(num_holes_range=(4, 4), hole_height_range=(0.08, 0.08),
                            hole_width_range=(0.08, 0.08), p=1.0),
        ], save_applied_params=True)),
        ("HFlip + Affine + Noise", A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Affine(scale=(1.1, 1.1), rotate=(-6, -6), balanced_scale=True,
                     mask_interpolation=cv2.INTER_NEAREST, p=1.0),
            A.GaussNoise(noise_scale_factor=0.8, p=1.0),
        ], save_applied_params=True)),
    ]

    seg_cells = []
    for i, (label, pipe) in enumerate(seg_variants):
        result = pipe(image=seg_base_img.copy(), mask=seg_base_mask.copy())
        applied = get_applied_names(result)
        vis = overlay_mask(result["image"], result["mask"])
        seg_cells.append(add_two_line_label(vis, f"Variant {i+1}", applied))

    pipelines = [
        ("Classification Pipeline", cls_cells, (40, 100, 50)),
        ("Object Detection Pipeline", det_cells, (40, 50, 120)),
        ("Segmentation Pipeline", seg_cells, (120, 60, 40)),
    ]
    rows = []
    for name, cells, hdr_color in pipelines:
        row = hstack_with_gap(cells, gap=4)
        rows.extend([section_header(row.shape[1], name, bg=hdr_color), row])
    save_webp(OUT_DIR / "pipeline_output_examples.webp",
              vstack_with_gap(rows, gap=6))


# ---------------------------------------------------------------------------
# Image 11: Crop vs Resize vs Letterbox
# ---------------------------------------------------------------------------
def build_crop_resize_letterbox(soccer: np.ndarray) -> None:
    print("Building crop_resize_letterbox...")
    target = 300
    h_orig, w_orig = soccer.shape[:2]

    orig_display = resize_height(soccer, target)
    if orig_display.shape[1] > 450:
        orig_display = orig_display[:, :450]

    np.random.seed(15)
    random.seed(15)
    cropped = A.Compose([
        A.RandomCrop(height=target, width=target, p=1.0),
    ])(image=soccer)["image"]

    shortest = A.Compose([
        A.SmallestMaxSize(max_size=target, p=1.0),
        A.CenterCrop(height=target, width=target, p=1.0),
    ])(image=soccer)["image"]

    letterboxed = A.Compose([
        A.LongestMaxSize(max_size=target, p=1.0),
        A.PadIfNeeded(min_height=target, min_width=target,
                      border_mode=cv2.BORDER_CONSTANT, fill=114, p=1.0),
    ])(image=soccer)["image"]

    cells = [
        add_two_line_label(orig_display, "Original",
                           f"{w_orig}x{h_orig}"),
        add_two_line_label(cropped, "RandomCrop",
                           f"Random {target}x{target} patch"),
        add_two_line_label(shortest, "Resize + CenterCrop",
                           "Shortest side first, then crop"),
        add_two_line_label(letterboxed, "Resize + Pad",
                           "Longest side first, then pad"),
    ]
    save_webp(OUT_DIR / "crop_resize_letterbox.webp", hstack_with_gap(cells))


# ---------------------------------------------------------------------------
# Image 12: Aggregate Metrics Hide Per-Class Regression
# ---------------------------------------------------------------------------
def build_aggregate_hides_regression() -> None:
    print("Building aggregate_hides_regression...")
    data = [
        ("Dog", +1.2),
        ("Cat", +0.8),
        ("Car", +0.3),
        ("Bird", +1.5),
        ("Flower", +0.9),
        ("Building", +0.4),
        ("Traffic Light", -5.2),
        ("Ripe Fruit", -8.1),
    ]
    agg_delta = 0.5

    W, ROW_H, LABEL_W, DELTA_W = 740, 34, 130, 65
    BAR_AREA_W = W - LABEL_W - DELTA_W
    HEADER_H = 44
    FOOTER_H = 40
    n = len(data)
    max_mag = max(abs(d) for _, d in data)
    total_h = HEADER_H + n * ROW_H + 6 + ROW_H + FOOTER_H
    canvas = np.full((total_h, W, 3), 255, dtype=np.uint8)
    center_x = LABEL_W + BAR_AREA_W // 2

    cv2.rectangle(canvas, (0, 0), (W, HEADER_H), (50, 50, 70), -1)
    cv2.putText(canvas, "Per-class accuracy change after adding ColorJitter",
                (10, 18), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "Aggregate improves +0.5%  --  but two color-dependent classes collapse",
                (10, 36), FONT, 0.35, (180, 180, 200), 1, cv2.LINE_AA)

    for i, (name, delta) in enumerate(data):
        y = HEADER_H + i * ROW_H
        if delta < -3:
            bg = (255, 230, 230)
        elif i % 2 == 0:
            bg = (245, 245, 245)
        else:
            bg = (255, 255, 255)
        cv2.rectangle(canvas, (0, y), (W, y + ROW_H), bg, -1)
        cv2.line(canvas, (0, y + ROW_H), (W, y + ROW_H), (220, 220, 220), 1)

        tc = (180, 40, 40) if delta < -3 else (50, 50, 50)
        cv2.putText(canvas, name, (8, y + ROW_H - 10), FONT, 0.40,
                    tc, 1, cv2.LINE_AA)

        bar_len = int(abs(delta) / max_mag * (BAR_AREA_W // 2 - 20))
        bar_color = (70, 170, 70) if delta >= 0 else (200, 55, 55)
        by0, by1 = y + 7, y + ROW_H - 7
        if delta >= 0:
            cv2.rectangle(canvas, (center_x, by0),
                          (center_x + bar_len, by1), bar_color, -1)
        else:
            cv2.rectangle(canvas, (center_x - bar_len, by0),
                          (center_x, by1), bar_color, -1)

        sign = "+" if delta >= 0 else ""
        cv2.putText(canvas, f"{sign}{delta}%",
                    (W - DELTA_W + 2, y + ROW_H - 10),
                    FONT, 0.38, tc, 1, cv2.LINE_AA)

    sep_y = HEADER_H + n * ROW_H
    cv2.line(canvas, (0, sep_y), (W, sep_y), (100, 100, 100), 2)

    agg_y = sep_y + 6
    cv2.rectangle(canvas, (0, agg_y), (W, agg_y + ROW_H),
                  (220, 235, 220), -1)
    cv2.putText(canvas, "AGGREGATE", (8, agg_y + ROW_H - 10), FONT, 0.45,
                (30, 30, 30), 1, cv2.LINE_AA)
    bar_len = int(agg_delta / max_mag * (BAR_AREA_W // 2 - 20))
    cv2.rectangle(canvas, (center_x, agg_y + 7),
                  (center_x + bar_len, agg_y + ROW_H - 7), (70, 170, 70), -1)
    cv2.putText(canvas, f"+{agg_delta}%",
                (W - DELTA_W + 2, agg_y + ROW_H - 10),
                FONT, 0.42, (70, 170, 70), 1, cv2.LINE_AA)

    cv2.line(canvas, (center_x, HEADER_H), (center_x, agg_y + ROW_H),
             (180, 180, 180), 1)

    footer_y = agg_y + ROW_H + 2
    cv2.rectangle(canvas, (0, footer_y), (W, footer_y + FOOTER_H),
                  (60, 40, 40), -1)
    cv2.putText(canvas,
                "Color-dependent classes lost their primary signal."
                " Aggregate hides the damage.",
                (8, footer_y + 16), FONT, 0.37, (255, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                "Always check per-class metrics before committing"
                " to a new augmentation.",
                (8, footer_y + 32), FONT, 0.35, (200, 180, 180), 1, cv2.LINE_AA)

    save_webp(OUT_DIR / "aggregate_hides_regression.webp", canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    random.seed(42)
    np.random.seed(42)

    fish = read_rgb(FISH_PATH)
    soccer = read_rgb(SOCCER_IMAGE_PATH)
    bboxes, labels, polygons = read_soccer_annotations(SOCCER_JSON_PATH)

    person_small = read_rgb(PERSON_SMALL_PATH)
    person_medium = read_rgb(PERSON_MEDIUM_PATH)
    person_large = read_rgb(PERSON_LARGE_PATH)

    build_header(fish)
    build_level1_vs_level2(fish)
    build_constrained_dropout(soccer, bboxes, labels)
    build_train_hard_test_easy(fish)
    build_pipeline_steps(fish)
    build_augmentation_bug(soccer, bboxes, labels)
    build_scale_variation(person_small, person_medium, person_large)
    build_digit_rotation()
    build_domain_transforms_sampler(fish)
    build_diagnostic_results_table()
    build_pipeline_output_examples(soccer, bboxes, labels, polygons)
    build_crop_resize_letterbox(soccer)
    build_aggregate_hides_regression()

    print("\nAll choosing-augmentations images generated.")


if __name__ == "__main__":
    main()
