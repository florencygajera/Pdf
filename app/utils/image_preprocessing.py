"""
Image Preprocessing Pipeline
Prepares scanned PDF page images for high-accuracy OCR using OpenCV.

Pipeline: grayscale → [optional: denoise] → adaptive-threshold → [optional: deskew] → morphology

CHANGES IN THIS VERSION (Indic script tuning):
  - estimate_page_complexity: dark_ratio / edge_ratio thresholds tightened
    so dense Gujarati text pages correctly trigger the full preprocessing path.
  - preprocess_page_image: contrast enhancement trigger raised (std < 50 not 40)
    because government notice scans frequently have slightly uneven contrast.
  - deskew() trigger threshold lowered (edge_ratio > 0.02 not 0.03) — Gujarati
    documents with table borders push edge_ratio just above 0.02.
  - remove_noise() uses medianBlur(3) — better salt-and-pepper removal than
    GaussianBlur while preserving character edges.
  - Warp uses INTER_LINEAR (faster, adequate quality for document text).
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from app.config.constants import (
    ADAPTIVE_BLOCK_SIZE,
    ADAPTIVE_C,
    DESKEW_MAX_ANGLE,
    MORPH_KERNEL_SIZE,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_page_complexity(image: np.ndarray) -> dict:
    """
    Cheap page-quality estimate used to decide whether a page needs the full
    preprocessing stack.

    Thresholds tuned for Gujarati government notice PDFs:
    - dark_ratio > 0.20 (was 0.28): dense Gujarati text + table borders
      push dark_ratio above 0.20 on clean pages — route them to full path.
    - edge_ratio > 0.08 (was 0.10): table lines + text edges in Gujarati
      notices raise edge_ratio; 0.08 catches these without over-triggering.
    - Canny skipped when std < 28 (near-blank pages) for speed.
    """
    gray = to_grayscale(image)
    h, w = gray.shape[:2]
    sample_w = min(384, w)
    sample_h = max(1, int(h * (sample_w / max(w, 1))))
    sample = cv2.resize(gray, (sample_w, sample_h), interpolation=cv2.INTER_AREA)

    std = float(np.std(sample))
    dark_ratio = float(np.mean(sample < 180))

    if std < 28.0:
        edge_ratio = 0.0
        needs_full_preprocess = True  # near-blank or uniform page
    else:
        edge_ratio = float(
            np.mean(cv2.Canny(sample, 50, 150) > 0) if sample.size else 0.0
        )
        # Tightened thresholds for Indic script docs
        needs_full_preprocess = dark_ratio > 0.20 or edge_ratio > 0.08

    return {
        "std": std,
        "dark_ratio": dark_ratio,
        "edge_ratio": edge_ratio,
        "needs_full_preprocess": needs_full_preprocess,
    }


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR to PIL Image RGB."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR to grayscale if not already."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """
    3×3 median blur — removes salt-and-pepper noise while preserving
    character edges better than GaussianBlur.
    """
    return cv2.medianBlur(gray, 3)


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding — handles uneven lighting common in
    scanned government documents.
    """
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def _collect_text_like_pixels(binary: np.ndarray) -> np.ndarray:
    """
    Collect dark pixels that belong to character-like connected components.
    Large horizontal rules, borders, and dense stamp blobs are filtered out.
    """
    dark_mask = (binary < 128).astype(np.uint8)
    if not np.any(dark_mask):
        return np.empty((0, 2), dtype=np.int32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dark_mask, connectivity=8
    )
    h, w = binary.shape
    total_area = h * w
    keep_labels = []

    for label in range(1, num_labels):
        x, y, comp_w, comp_h, area = stats[label]
        if area < 12:
            continue
        if area > total_area * 0.08:
            continue
        bbox_area = max(1, comp_w * comp_h)
        fill_ratio = area / bbox_area
        if (comp_h <= 2 and comp_w > w * 0.25) or (comp_w <= 2 and comp_h > h * 0.25):
            continue
        if fill_ratio < 0.03 and max(comp_w, comp_h) > max(w, h) * 0.2:
            continue
        keep_labels.append(label)

    if not keep_labels:
        return np.empty((0, 2), dtype=np.int32)

    selected = np.isin(labels, keep_labels)
    return np.column_stack(np.where(selected))


def deskew(binary: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Detect and correct rotation angle using character-like connected components.
    Returns (corrected_image, angle_degrees).
    """
    coords = _collect_text_like_pixels(binary)
    if coords.size == 0:
        coords = np.column_stack(np.where(binary < 128))
        if coords.size == 0:
            return binary, 0.0

    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > DESKEW_MAX_ANGLE:
        logger.warning("Deskew angle %.1f° exceeds threshold, skipping.", angle)
        return binary, angle

    if abs(angle) < 0.3:
        return binary, 0.0

    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskewed by %.2f°", angle)
    return rotated, angle


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Morphological opening removes tiny noise dots.
    Closing fills small gaps within characters.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def remove_stamp_artifacts(binary: np.ndarray) -> np.ndarray:
    """
    Detect and blank large filled blobs (stamps, seals).
    Only runs in the full preprocessing path.
    """
    h, w = binary.shape
    total_area = h * w
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned = binary.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.02:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if solidity < 0.5:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
            logger.debug(
                "Removed stamp artifact (area=%.0f, solidity=%.2f)", area, solidity
            )
    return cleaned


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """CLAHE contrast enhancement — improves faded/low-contrast scans."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def preprocess_page_image(
    image: Image.Image,
    apply_stamp_removal: bool = True,
    apply_deskew: bool = True,
    prefer_light: bool = False,
    quality_profile: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for a single PDF page image.

    Light path (clean / standard pages):
      grayscale → adaptive threshold → [optional deskew] → morphological cleanup

    Full path (noisy / low-contrast / heavy pages):
      grayscale → [contrast enhance if std < 50] → [denoise if very dark]
      → adaptive threshold → [stamp removal] → [deskew] → morphological cleanup

    Tuning changes for Indic script:
    - contrast enhancement triggered at std < 50 (was 40) — government
      notice scans often have slightly uneven contrast from flatbed scanners.
    - deskew triggered at edge_ratio > 0.02 (was 0.03) — Gujarati docs with
      table borders hover just above 0.02 and benefit from deskew.

    Returns:
        (processed_ndarray_BGR, metadata_dict)
    """
    meta: dict = {}
    img_cv = pil_to_cv2(image)

    gray = to_grayscale(img_cv)
    meta["original_shape"] = gray.shape

    quality = quality_profile or estimate_page_complexity(gray)
    meta["quality"] = quality

    use_light_path = prefer_light or not quality["needs_full_preprocess"]
    meta["preprocess_mode"] = "light" if use_light_path else "full"

    if use_light_path:
        binary = adaptive_threshold(gray)
        angle = 0.0
        # Deskew if edges suggest skew — threshold 0.02 catches table-border pages
        if apply_deskew and quality["edge_ratio"] > 0.02:
            binary, angle = deskew(binary)
        meta["deskew_angle"] = angle
        binary = morphological_cleanup(binary)
    else:
        # Contrast enhancement — trigger at std < 50 for Indic script scans
        if quality["std"] < 50.0:
            gray = enhance_contrast(gray)

        if quality["std"] < 20.0 or quality["dark_ratio"] > 0.35:
            gray = remove_noise(gray)

        binary = adaptive_threshold(gray)

        if apply_stamp_removal:
            binary = remove_stamp_artifacts(binary)

        angle = 0.0
        if apply_deskew:
            binary, angle = deskew(binary)
        meta["deskew_angle"] = angle
        binary = morphological_cleanup(binary)

    # Convert back to 3-channel BGR for PaddleOCR
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return output, meta
