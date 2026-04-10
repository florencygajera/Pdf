"""
Image Preprocessing Pipeline
Prepares scanned PDF page images for high-accuracy OCR using OpenCV.

Pipeline: grayscale → denoise → adaptive-threshold → deskew → morphology
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
    Apply Non-Local Means Denoising — effective for printed document scans.
    h=10: filter strength (higher = more denoising, less detail).
    """
    return cv2.fastNlMeansDenoising(
        gray, h=10, templateWindowSize=7, searchWindowSize=21
    )


def adaptive_threshold(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.
    Handles uneven lighting common in scanned government docs.
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

    Large horizontal rules, borders, and dense stamp blobs are filtered out so
    they do not dominate the skew estimate.
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

        # Long, thin components are usually rules, borders, or table lines.
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

    # Normalize angle: OpenCV returns -90..0, we want -45..45
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) > DESKEW_MAX_ANGLE:
        logger.warning(
            f"Deskew angle {angle:.1f}° exceeds threshold, skipping rotation."
        )
        return binary, angle

    if abs(angle) < 0.1:
        return binary, 0.0  # No meaningful rotation

    # Rotate the image
    h, w = binary.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug(f"Deskewed by {angle:.2f}°")
    return rotated, angle


def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Morphological opening removes tiny noise dots.
    Closing fills small gaps within characters.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_KERNEL_SIZE)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return opened


def remove_stamp_artifacts(binary: np.ndarray) -> np.ndarray:
    """
    Detect and blank large filled blobs (stamps, seals).
    Strategy: find contours with large area but irregular shape,
    fill them white.
    """
    h, w = binary.shape
    total_area = h * w
    inverted = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cleaned = binary.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Stamps are large blobs (>2% of image) with low solidity
        if area < total_area * 0.02:
            continue
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            continue
        solidity = area / hull_area
        # Text contours have higher solidity; stamps are more irregular
        if solidity < 0.5:
            cv2.drawContours(cleaned, [cnt], -1, 255, thickness=cv2.FILLED)
            logger.debug(
                f"Removed possible stamp artifact (area={area:.0f}, solidity={solidity:.2f})"
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
) -> Tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline for a single PDF page image.

    Args:
        image: PIL Image from pdf2image
        apply_stamp_removal: remove stamp-like blobs
        apply_deskew: correct skewed scans

    Returns:
        (processed_ndarray, metadata_dict)
    """
    meta: dict = {}
    img_cv = pil_to_cv2(image)

    # 1. Grayscale
    gray = to_grayscale(img_cv)
    meta["original_shape"] = gray.shape

    # 2. Contrast enhancement
    gray = enhance_contrast(gray)

    # 3. Denoise
    gray = remove_noise(gray)

    # 4. Adaptive threshold
    binary = adaptive_threshold(gray)

    # 5. Stamp removal
    if apply_stamp_removal:
        binary = remove_stamp_artifacts(binary)

    # 6. Deskew
    angle = 0.0
    if apply_deskew:
        binary, angle = deskew(binary)
    meta["deskew_angle"] = angle

    # 7. Morphological cleanup
    binary = morphological_cleanup(binary)

    # Convert back to 3-channel for PaddleOCR (expects BGR)
    output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return output, meta
