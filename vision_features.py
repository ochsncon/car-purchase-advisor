"""Feature extraction helpers for the local vehicle image classifier."""

from __future__ import annotations

import numpy as np
from PIL import Image


def ensure_pil_image(image: Image.Image | np.ndarray | None) -> Image.Image:
    if image is None:
        raise ValueError("No image provided.")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image.astype("uint8"))
    if not isinstance(image, Image.Image):
        raise TypeError("Input is not a valid image.")
    return image


def extract_image_features(image: Image.Image | np.ndarray | None) -> np.ndarray:
    """Extract rich feature set from car images using multiple methods.

    Combines:
    - Color statistics (RGB/HSV/LAB mean/std, histograms)
    - HOG (Histogram of Oriented Gradients) for shape/edge info
    - LBP (Local Binary Pattern) for texture
    - Spatial grid features (multi-scale)
    - Edge density and contrast
    """
    pil_image = ensure_pil_image(image).convert("RGB").resize((128, 128))
    arr = np.asarray(pil_image, dtype=np.float32) / 255.0

    # === Color Features ===
    # RGB statistics
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))

    # Per-channel histograms (16 bins each = better granularity)
    color_hist_features = []
    for channel_index in range(3):
        ch_hist, _ = np.histogram(arr[:, :, channel_index], bins=16, range=(0.0, 1.0), density=True)
        color_hist_features.append(ch_hist.astype(np.float32))
    color_hist = np.concatenate(color_hist_features)

    # Saturation & Value channels (approximated from RGB)
    # S = (max - min) / max, V = max
    rgb_max = arr.max(axis=2)
    rgb_min = arr.min(axis=2)
    saturation = (rgb_max - rgb_min) / np.clip(rgb_max, 1e-6, 1.0)
    value = rgb_max
    s_mean = float(saturation.mean())
    s_std = float(saturation.std())
    v_mean = float(value.mean())
    v_std = float(value.std())

    # Grayscale features
    gray = arr.mean(axis=2)
    gray_hist, _ = np.histogram(gray, bins=32, range=(0.0, 1.0), density=True)

    # === Edge & Gradient Features (simplified HOG) ===
    grad_x = np.gradient(gray, axis=1)
    grad_y = np.gradient(gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)

    # Quantized gradient direction histogram (8 bins)
    grad_hist, _ = np.histogram(grad_dir, bins=8, range=(-np.pi, np.pi), density=True)
    grad_mag_mean = float(grad_mag.mean())
    grad_mag_std = float(grad_mag.std())

    # === Texture Features (LBP-inspired) ===
    # Local Binary Pattern: compare each pixel with 8 neighbors
    lbp_features = _compute_lbp_features(gray)

    # === Spatial Features (multi-scale grids) ===
    # Coarse grids at different scales for spatial information
    coarse_4x4 = np.asarray(pil_image.convert("L").resize((4, 4)), dtype=np.float32) / 255.0
    coarse_8x8 = np.asarray(pil_image.convert("L").resize((8, 8)), dtype=np.float32) / 255.0
    coarse_16x16 = np.asarray(pil_image.convert("L").resize((16, 16)), dtype=np.float32) / 255.0

    # === Contrast & Edge Density ===
    contrast = float(gray.std())
    edge_density = float(np.mean(grad_mag > 0.1))  # Proportion of "edgy" pixels

    # Concatenate all features
    features = np.concatenate([
        rgb_mean,
        rgb_std,
        color_hist,
        np.array([s_mean, s_std, v_mean, v_std], dtype=np.float32),
        gray_hist.astype(np.float32),
        grad_hist.astype(np.float32),
        np.array([grad_mag_mean, grad_mag_std], dtype=np.float32),
        lbp_features.astype(np.float32),
        coarse_4x4.flatten().astype(np.float32),
        coarse_8x8.flatten().astype(np.float32),
        coarse_16x16.flatten().astype(np.float32),
        np.array([contrast, edge_density], dtype=np.float32),
    ])
    return features


def _compute_lbp_features(gray: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Compute simplified LBP (Local Binary Pattern) histogram.
    
    Compare each pixel with its 8 neighbors to extract local texture patterns.
    """
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # Simplified LBP: compare with mean of neighbors
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            neighbors = gray[i-1:i+2, j-1:j+2].flatten()
            # Count neighbors brighter than center (0-8 possible)
            lbp[i, j] = np.sum(neighbors > center)

    # Histogram of LBP values (0-8 range)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, 9), density=True)
    return hist