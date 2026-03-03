from __future__ import annotations

import json
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PARROT_PATH = ROOT / "temp_img" / "parrot.webp"
ROAD_IMAGE_PATH = ROOT / "temp_img" / "road.webp"
ROAD_JSON_PATH = ROOT / "temp_img" / "road.json"

INTRO_DIR = ROOT / "img" / "introduction" / "what-are-image-augmentations"
COLLAGE_DIR = ROOT / "img" / "introduction" / "image_augmentation"

CANVAS_SIZE = 384
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_webp(path: Path, image_rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def normalize_for_grid(image_rgb: np.ndarray, size: int = CANVAS_SIZE) -> np.ndarray:
    transform = A.Compose(
        [
            A.LongestMaxSize(max_size=size, p=1.0),
            A.PadIfNeeded(
                min_height=size,
                min_width=size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=0,
                p=1.0,
            ),
        ]
    )
    return transform(image=image_rgb)["image"]


def apply_transform(image_rgb: np.ndarray, transform: A.BasicTransform) -> np.ndarray:
    return transform(image=image_rgb)["image"]


def draw_label(image_rgb: np.ndarray, text: str) -> np.ndarray:
    out = image_rgb.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_grid(images: list[np.ndarray], labels: list[str], cols: int, padding: int = 16) -> np.ndarray:
    if len(images) != len(labels):
        raise ValueError("Images and labels must have same length.")
    if not images:
        raise ValueError("No images provided.")

    h, w = images[0].shape[:2]
    for image in images:
        if image.shape[:2] != (h, w):
            raise ValueError("All images must have same shape.")

    rows = (len(images) + cols - 1) // cols
    grid_h = rows * h + (rows + 1) * padding
    grid_w = cols * w + (cols + 1) * padding
    canvas = np.full((grid_h, grid_w, 3), 245, dtype=np.uint8)

    for idx, (image, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols
        y = padding + row * (h + padding)
        x = padding + col * (w + padding)
        labeled = draw_label(image, label)
        canvas[y : y + h, x : x + w] = labeled

    return canvas


def load_labelme_polygons(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    shapes = data.get("shapes", [])
    polygons = []
    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        points_arr = np.array(points, dtype=np.float32)
        polygons.append(
            {
                "label": str(shape.get("label", "")),
                "points": points_arr,
            }
        )
    return polygons


def polygon_to_bbox(points: np.ndarray) -> tuple[float, float, float, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def polygons_to_mask(image_shape: tuple[int, int, int], polygons: list[dict], label: str) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        if polygon["label"] != label:
            continue
        pts = np.round(polygon["points"]).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=1)
    return mask


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    out = image_rgb.copy()
    color_arr = np.zeros_like(out)
    color_arr[:, :] = np.array(color, dtype=np.uint8)
    alpha = 0.45
    mask_bool = mask.astype(bool)
    out[mask_bool] = (alpha * color_arr[mask_bool] + (1 - alpha) * out[mask_bool]).astype(np.uint8)
    return out


def draw_bboxes(image_rgb: np.ndarray, bboxes: list[tuple[float, float, float, float]], color: tuple[int, int, int]) -> np.ndarray:
    out = image_rgb.copy()
    for x_min, y_min, x_max, y_max in bboxes:
        p1 = (int(round(x_min)), int(round(y_min)))
        p2 = (int(round(x_max)), int(round(y_max)))
        cv2.rectangle(out, p1, p2, color, 2)
    return out


def draw_keypoints(image_rgb: np.ndarray, keypoints: list[tuple[float, float]], color: tuple[int, int, int]) -> np.ndarray:
    out = image_rgb.copy()
    for x, y in keypoints:
        center = (int(round(x)), int(round(y)))
        cv2.circle(out, center, radius=4, color=color, thickness=-1)
    return out


def make_tight_grid(images: list[np.ndarray], cols: int) -> np.ndarray:
    """Pack images into a grid with no padding and no labels."""
    if not images:
        raise ValueError("No images provided.")
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, image in enumerate(images):
        r = idx // cols
        c = idx % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = image
    return canvas


HEADER_COLS = 8
HEADER_ROWS = 5
HEADER_CELL_SIZE = 160

# One deterministic transform per grid cell, showcasing Albumentations breadth.
# Every transform gets a random D4 applied on top for orientation variety.
def _header_transforms(size: int) -> list[A.BasicTransform]:
    return [
        # --- Geometry ---
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=(25, 25), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Rotate(limit=(-40, -40), border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Affine(scale=0.65, rotate=0, shear=0, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Affine(scale=1.0, rotate=0, shear=25, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.Perspective(scale=0.15, border_mode=cv2.BORDER_CONSTANT, p=1.0),
        A.ElasticTransform(alpha=120, sigma=6, p=1.0),
        A.GridDistortion(num_steps=5, distort_limit=0.4, p=1.0),
        A.OpticalDistortion(distort_limit=0.5, p=1.0),
        A.ThinPlateSpline(scale_range=(0.12, 0.12), p=1.0),
        A.RandomResizedCrop(size=(size, size), scale=(0.35, 0.45), ratio=(0.9, 1.1), p=1.0),
        A.RandomGridShuffle(grid=(3, 3), p=1.0),
        # --- Color / photometric ---
        A.RandomBrightnessContrast(brightness_limit=(0.45, 0.45), contrast_limit=(0.4, 0.4), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.25), contrast_limit=(-0.15, -0.15), p=1.0),
        A.PhotoMetricDistort(p=1.0),
        A.PhotoMetricDistort(p=1.0),
        A.RandomGamma(gamma_limit=(35, 35), p=1.0),
        A.RandomGamma(gamma_limit=(200, 200), p=1.0),
        A.CLAHE(clip_limit=6.0, p=1.0),
        A.ColorJitter(brightness=0.0, contrast=0.0, saturation=2.0, hue=0.2, p=1.0),
        A.ToGray(p=1.0),
        A.ToSepia(p=1.0),
        A.Posterize(num_bits=2, p=1.0),
        A.Equalize(p=1.0),
        A.Solarize(threshold_range=(0.45, 0.55), p=1.0),
        A.ChannelShuffle(p=1.0),
        A.ChannelDropout(channel_drop_range=(1, 1), p=1.0),
        A.RandomToneCurve(scale=0.4, p=1.0),
        A.PlanckianJitter(mode="blackbody", p=1.0),
        A.AutoContrast(p=1.0),
        A.FancyPCA(alpha=0.6, p=1.0),
        A.PlasmaBrightnessContrast(p=1.0),
        A.Illumination(p=1.0),
        # --- Blur / noise / sharpness ---
        A.GaussianBlur(blur_limit=(13, 13), p=1.0),
        A.MotionBlur(blur_limit=(19, 19), p=1.0),
        A.MedianBlur(blur_limit=(11, 11), p=1.0),
        A.GlassBlur(sigma=0.7, max_delta=3, iterations=2, p=1.0),
        A.ZoomBlur(max_factor=1.2, p=1.0),
        A.Defocus(radius=(6, 6), alias_blur=0.1, p=1.0),
        A.AdvancedBlur(blur_limit=(9, 13), p=1.0),
        A.GaussNoise(std_range=(0.12, 0.14), p=1.0),
        A.ISONoise(color_shift=(0.1, 0.3), intensity=(0.7, 0.9), p=1.0),
        A.ShotNoise(scale_range=(0.18, 0.22), p=1.0),
        A.SaltAndPepper(amount=(0.06, 0.08), p=1.0),
        A.MultiplicativeNoise(multiplier=(0.55, 0.55), p=1.0),
        A.Sharpen(alpha=(0.9, 0.9), lightness=(1.0, 1.0), p=1.0),
        A.UnsharpMask(blur_limit=(7, 7), sigma_limit=0, alpha=(0.8, 0.8), p=1.0),
        A.Emboss(alpha=(0.8, 0.8), strength=(1.2, 1.2), p=1.0),
        A.RingingOvershoot(blur_limit=(7, 9), p=1.0),
        A.Superpixels(p_replace=0.6, n_segments=80, p=1.0),
        A.Downscale(scale_range=(0.2, 0.2), p=1.0),
        A.ImageCompression(quality_range=(5, 8), p=1.0),
        A.ChromaticAberration(primary_distortion_limit=0.07, secondary_distortion_limit=0.07, p=1.0),
        # --- Occlusion / masking ---
        A.CoarseDropout(num_holes_range=(8, 10), hole_height_range=(0.07, 0.1), hole_width_range=(0.07, 0.1), fill=0, p=1.0),
        A.CoarseDropout(num_holes_range=(8, 10), hole_height_range=(0.07, 0.1), hole_width_range=(0.07, 0.1), fill="random", p=1.0),
        A.GridDropout(ratio=0.45, p=1.0),
        A.PixelDropout(dropout_prob=0.2, p=1.0),
        # --- Weather / environment / misc ---
        A.Dithering(p=1.0),
        A.Spatter(mean=0.65, std=0.3, gauss_sigma=2, p=1.0),
        A.RandomFog(fog_coef_range=(0.5, 0.7), p=1.0),
        A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1, p=1.0),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1.0),
        A.RandomSunFlare(flare_roi=(0.2, 0.2, 0.8, 0.8), p=1.0),
        A.RandomSnow(snow_point_range=(0.3, 0.4), brightness_coeff=2.0, p=1.0),
        A.RandomGravel(gravel_roi=(0, 0.4, 1, 1), number_of_patches=2, p=1.0),
        A.PlasmaShadow(p=1.0),
        A.AdditiveNoise(noise_type="uniform", spatial_mode="per_pixel", p=1.0),
    ]


def build_header_image(parrot_rgb: np.ndarray) -> None:
    size = HEADER_CELL_SIZE
    # Square-crop from center so there's no black padding in the base image.
    # Geometry transforms that rotate will still show black corners, which is fine —
    # that's exactly what the transform looks like.
    resize = A.Compose([
        A.SmallestMaxSize(max_size=size, p=1.0),
        A.CenterCrop(height=size, width=size, p=1.0),
    ])
    base = resize(image=parrot_rgb)["image"]

    d4 = A.D4(p=1.0)
    all_transforms = _header_transforms(size)
    n = HEADER_COLS * HEADER_ROWS

    images = []
    for t in all_transforms[:n]:
        img = apply_transform(base, t)
        img = apply_transform(img, d4)
        images.append(img)

    grid = make_tight_grid(images, cols=HEADER_COLS)
    save_webp(INTRO_DIR / "header_augmentation_mosaic.webp", grid)


def build_parrot_images(parrot_rgb: np.ndarray) -> None:
    base = normalize_for_grid(parrot_rgb)

    collage_ops = [
        ("Original", A.NoOp(p=1.0)),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("RandomCrop", A.RandomResizedCrop(size=(CANVAS_SIZE, CANVAS_SIZE), scale=(0.4, 0.55), ratio=(0.9, 1.1), p=1.0)),
        ("MedianBlur", A.MedianBlur(blur_limit=(15, 15), p=1.0)),
        ("BrightnessContrast", A.RandomBrightnessContrast(brightness_limit=(0.3, 0.3), contrast_limit=(0.25, 0.25), p=1.0)),
        ("PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
        ("Gamma", A.RandomGamma(gamma_limit=(70, 70), p=1.0)),
        ("GaussNoise", A.GaussNoise(std_range=(0.15, 0.15), p=1.0)),
        ("MotionBlur", A.MotionBlur(blur_limit=(11, 11), p=1.0)),
        (
            "AffineRotScale",
            A.Affine(
                scale=(0.8, 1.2),
                rotate=(-25, 25),
                shear=(0.0, 0.0),
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
        ),
    ]

    collage_images = [apply_transform(base, op) for _, op in collage_ops]
    collage_labels = [name for name, _ in collage_ops]
    collage = make_grid(collage_images, collage_labels, cols=5)
    save_webp(COLLAGE_DIR / "augmentation.webp", collage)

    label_preservation_ops = [
        ("Original (Label: parrot)", A.NoOp(p=1.0)),
        ("Flip (Label: parrot)", A.HorizontalFlip(p=1.0)),
        ("Rotate +/-12deg", A.Rotate(limit=(-12, 12), border_mode=cv2.BORDER_CONSTANT, p=1.0)),
        ("Light/Contrast", A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.15, 0.15), p=1.0)),
        ("Aggressive Blur", A.GaussianBlur(blur_limit=(11, 11), p=1.0)),
        ("PhotoMetricDistort", A.PhotoMetricDistort(p=1.0)),
    ]
    label_images = [apply_transform(base, op) for _, op in label_preservation_ops]
    label_labels = [name for name, _ in label_preservation_ops]
    label_grid = make_grid(label_images, label_labels, cols=3)
    save_webp(INTRO_DIR / "parrot_label_preservation.webp", label_grid)

    over_ops = [
        ("Original", A.NoOp(p=1.0)),
        (
            "Realistic Policy",
            A.Compose(
                [
                    A.HorizontalFlip(p=1.0),
                    A.Rotate(limit=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(0.15, 0.15), contrast_limit=(0.15, 0.15), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 3), p=1.0),
                ]
            ),
        ),
        (
            "Over-Augmented",
            A.Compose(
                [
                    A.Rotate(limit=(80, 80), border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
                    A.RandomBrightnessContrast(brightness_limit=(0.8, 0.8), contrast_limit=(0.8, 0.8), p=1.0),
                    A.ColorJitter(brightness=0.5, contrast=0.5, saturation=2.0, hue=0.4, p=1.0),
                    A.CoarseDropout(
                        num_holes_range=(10, 10),
                        hole_height_range=(0.08, 0.16),
                        hole_width_range=(0.08, 0.16),
                        fill=0,
                        p=1.0,
                    ),
                ]
            ),
        ),
    ]
    over_images = [apply_transform(base, op) for _, op in over_ops]
    over_labels = [name for name, _ in over_ops]
    over_grid = make_grid(over_images, over_labels, cols=3)
    save_webp(INTRO_DIR / "over_augmentation.webp", over_grid)

    starter_ops = [
        ("Input", A.NoOp(p=1.0)),
        ("RandomResizedCrop", A.RandomResizedCrop(size=(CANVAS_SIZE, CANVAS_SIZE), scale=(0.85, 0.95), ratio=(0.9, 1.1), p=1.0)),
        ("HorizontalFlip", A.HorizontalFlip(p=1.0)),
        ("Rotate(limit=10)", A.Rotate(limit=(10, 10), border_mode=cv2.BORDER_CONSTANT, p=1.0)),
        ("BrightnessContrast", A.RandomBrightnessContrast(brightness_limit=(0.2, 0.2), contrast_limit=(0.2, 0.2), p=1.0)),
    ]
    starter_images = [apply_transform(base, op) for _, op in starter_ops]
    starter_labels = [name for name, _ in starter_ops]

    parrot_bbox = [0.3, 0.8, 0.8, 1.0]
    constrained_dropout = A.Compose(
        [A.ConstrainedCoarseDropout(num_holes_range=(6, 6), hole_height_range=(0.15, 0.3), hole_width_range=(0.15, 0.3), fill=0, bbox_labels=["parrot"], p=1.0)],
        bbox_params=A.BboxParams(coord_format="albumentations", label_fields=["labels"]),
    )
    dropout_result = constrained_dropout(image=base, bboxes=[parrot_bbox], labels=["parrot"])
    starter_images.append(dropout_result["image"])
    starter_labels.append("ConstrainedDropout")
    starter_grid = make_grid(starter_images, starter_labels, cols=3)
    save_webp(INTRO_DIR / "starter_policy_preview.webp", starter_grid)


def build_sync_figure(road_rgb: np.ndarray, polygons: list[dict]) -> None:
    car_polygons = [poly for poly in polygons if poly["label"].lower() == "car"]
    car_mask = polygons_to_mask(road_rgb.shape, car_polygons, label="car")
    bboxes = [polygon_to_bbox(poly["points"]) for poly in car_polygons]
    bbox_labels = ["car"] * len(bboxes)
    keypoints = [((x_min + x_max) * 0.5, (y_min + y_max) * 0.5) for x_min, y_min, x_max, y_max in bboxes]
    keypoint_labels = ["car_center"] * len(keypoints)

    pixel_transform = A.Compose(
        [A.PhotoMetricDistort(p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"], remove_invisible=False),
    )
    pixel_out = pixel_transform(
        image=road_rgb,
        mask=car_mask,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
    )

    spatial_transform = A.Compose(
        [A.Rotate(limit=(25, 25), border_mode=cv2.BORDER_CONSTANT, p=1.0)],
        bbox_params=A.BboxParams(coord_format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(coord_format="xy", label_fields=["keypoint_labels"], remove_invisible=False),
    )
    spatial_out = spatial_transform(
        image=road_rgb,
        mask=car_mask,
        bboxes=bboxes,
        bbox_labels=bbox_labels,
        keypoints=keypoints,
        keypoint_labels=keypoint_labels,
    )

    panels = [
        overlay_mask(pixel_out["image"], pixel_out["mask"], color=(255, 50, 50)),
        overlay_mask(spatial_out["image"], spatial_out["mask"], color=(255, 50, 50)),
        draw_keypoints(draw_bboxes(pixel_out["image"], pixel_out["bboxes"], color=(0, 255, 0)), pixel_out["keypoints"], color=(255, 0, 255)),
        draw_keypoints(
            draw_bboxes(spatial_out["image"], spatial_out["bboxes"], color=(0, 255, 0)),
            spatial_out["keypoints"],
            color=(255, 0, 255),
        ),
    ]
    panel_labels = [
        "Pixel Transform + Mask Overlay",
        "Spatial Transform + Mask Overlay",
        "Pixel Transform + BBoxes",
        "Spatial Transform + BBoxes",
    ]

    normalized = [normalize_for_grid(panel) for panel in panels]
    grid = make_grid(normalized, panel_labels, cols=2)
    save_webp(INTRO_DIR / "target_sync_road_mask_bbox.webp", grid)


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    parrot_rgb = read_rgb(PARROT_PATH)
    build_header_image(parrot_rgb)
    build_parrot_images(parrot_rgb)

    road_rgb = read_rgb(ROAD_IMAGE_PATH)
    polygons = load_labelme_polygons(ROAD_JSON_PATH)
    build_sync_figure(road_rgb, polygons)

    print("Generated intro augmentation assets successfully.")


if __name__ == "__main__":
    main()
