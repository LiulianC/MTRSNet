#!/usr/bin/env python3
"""
Generate pseudo reflection masks for benchmark construction.

The script follows the same high-level procedure used by the released data:
  input image -> grayscale -> valid FOV mask -> blurred baseline
  -> double-divided grayscale map -> thresholded pseudo reflection mask

Outputs:
  *_dd_grey.png  : double-divided grayscale map stored as uint16 PNG
  *_ref_mask.png : pseudo reflection mask, 255=reflection, 0=non-reflection

Usage:
  python delight.py <image_or_dir> [divisor=10] [erode_px=3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


EPSILON = 1e-8
BLACK_THR = 0.05
DEFAULT_DIVISOR = 10
DEFAULT_ERODE = 3
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_SUFFIXES = ("_dd_grey.png", "_ref_mask.png", "_dark_mask.png", "_dark_mask_sq.png")


def _kernel(width: int, divisor: int) -> int:
    k = max(3, width // max(1, divisor))
    if k % 2 == 0:
        k += 1
    return k


def _border_mask(grey: np.ndarray, thr: float = BLACK_THR, erode_px: int = DEFAULT_ERODE) -> np.ndarray:
    """Return a 0/1 mask where 1 marks the valid field of view."""
    h, w = grey.shape
    raw = (grey < thr).astype(np.uint8)
    num, labels = cv2.connectedComponents(raw, connectivity=4)
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    keep = set()
    for lbl in range(1, num):
        comp = labels == lbl
        if any(comp[y, x] for y, x in corners):
            keep.add(lbl)

    mask = np.zeros_like(raw, dtype=np.uint8)
    for lbl in keep:
        mask[labels == lbl] = 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    if erode_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        mask = cv2.dilate(mask, k2, iterations=1)
    return 1.0 - mask.astype(np.float32) / 255.0


def _dark_mask(v: np.ndarray, dilate_radius: int = 3, square: bool = False) -> np.ndarray:
    """Threshold the double-divided grayscale map and optionally dilate it."""
    v8 = np.clip(v * 255.0, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(v8, 254, 255, cv2.THRESH_BINARY_INV)
    if dilate_radius > 0:
        k = dilate_radius * 2 + 1
        if square:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.dilate(bw, kernel, iterations=1)
    return bw


def delight(img, divisor: int = DEFAULT_DIVISOR, erode_px: int = DEFAULT_ERODE, color_order: str = "BGR"):
    """
    Return the double-divided grayscale map and a pseudo reflection mask.

    Args:
        img: image path or numpy array
        divisor: Gaussian kernel divisor
        erode_px: extra dilation applied to the detected border mask
        color_order: channel order for numpy array input, either "BGR" or "RGB"
    """
    if isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
        if img is None:
            raise ValueError(f"Cannot read image: {img}")
        color_order = "BGR"

    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    if img.ndim == 2:
        raw = np.repeat(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 3:
        if color_order.upper() == "BGR":
            raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_order.upper() == "RGB":
            raw = img
        else:
            raise ValueError(f"Unsupported color_order: {color_order}")
    else:
        raw = img

    width = raw.shape[1]
    kernel = _kernel(width, divisor)

    grey = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    fov_mask = _border_mask(grey, erode_px=erode_px)
    grey_mask = grey * fov_mask + 1.0 * (1.0 - fov_mask)
    blur = cv2.GaussianBlur(grey_mask, (kernel, kernel), 0)
    delight_grey = np.clip(blur / np.clip(grey, EPSILON, None), 0, 1)
    double_div_grey = np.clip(delight_grey / np.clip(grey, EPSILON, None), 0, 1)

    ref_mask = _dark_mask(double_div_grey, dilate_radius=1, square=True)
    normal_mask = (fov_mask > 0.5).astype(np.uint8) * 255
    return double_div_grey, ref_mask, normal_mask


def reflection_stats(ref_mask: np.ndarray, normal_mask: np.ndarray):
    """Return reflection area, normal-FOV area, and their ratio."""
    ref = (ref_mask > 127) & (normal_mask > 127)
    normal = normal_mask > 127
    ref_area = int(ref.sum())
    normal_area = int(normal.sum())
    ratio = ref_area / normal_area if normal_area > 0 else 0.0
    return ref_area, normal_area, ratio


def iter_image_files(path: Path):
    if path.is_file():
        yield path
        return

    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if any(candidate.name.endswith(suffix) for suffix in SKIP_SUFFIXES):
            continue
        yield candidate


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    if len(argv) < 2:
        print("Usage: python delight.py <image_or_dir> [divisor=10] [erode_px=3]")
        return 1

    source = Path(argv[1])
    divisor = int(argv[2]) if len(argv) > 2 else DEFAULT_DIVISOR
    erode_px = int(argv[3]) if len(argv) > 3 else DEFAULT_ERODE

    if not source.exists():
        print(f"ERROR: path does not exist: {source}", file=sys.stderr)
        return 2

    for file_path in iter_image_files(source):
        try:
            result, ref_mask, normal_mask = delight(file_path, divisor=divisor, erode_px=erode_px)
        except Exception as exc:  # pragma: no cover - CLI diagnostic
            print(f"SKIP {file_path}: {exc}")
            continue

        out_dd = file_path.with_name(f"{file_path.stem}_dd_grey.png")
        out_ref = file_path.with_name(f"{file_path.stem}_ref_mask.png")
        cv2.imwrite(str(out_dd), (np.clip(result * 65535.0, 0, 65535)).astype(np.uint16))
        cv2.imwrite(str(out_ref), ref_mask)

        ref_area, normal_area, ratio = reflection_stats(ref_mask, normal_mask)
        pct = ref_area / ref_mask.size * 100.0
        print(
            f"{file_path.name}: ref_area={ref_area:>6d}px ({pct:.2f}%)  "
            f"normal_area={normal_area:>6d}px  "
            f"dark/normal={ratio:.4f}  ({ratio * 100:.2f}% of FOV)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
