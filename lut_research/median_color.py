"""
median_color.py

Utility for computing the perceptual median color of a masked region in an RGB image.
Used for color analysis in segmentation-based color transfer pipelines.
"""

import numpy as np
from skimage.color import rgb2lab, lab2rgb

def median_color(image, mask):
    """
    Compute the perceptual median color of the pixels in 'image' where mask == True.

    Args:
        image: H x W x 3 numpy array (RGB, 0-255)
        mask:  H x W boolean or int array (True/1 for pixels to include)

    Returns:
        Median color as an (R, G, B) tuple (0-255, uint8)
        If no pixels are selected, returns (0, 0, 0).
    """
    # Select pixels where mask is True
    pixels = image[mask]
    if len(pixels) == 0:
        return (0, 0, 0)
    # Convert to LAB color space for perceptual median calculation
    # Reshape for skimage: (N, 1, 3), scale to [0,1]
    pixels_lab = rgb2lab(pixels.reshape(-1, 1, 3) / 255.0).reshape(-1, 3)
    # Compute median in LAB space (perceptually uniform)
    median_lab = np.median(pixels_lab, axis=0)
    # Convert back to RGB, scale to [0,255], and clip
    median_rgb = lab2rgb(median_lab.reshape(1, 1, 3)).reshape(3)
    median_rgb = np.clip(median_rgb * 255, 0, 255).astype(np.uint8)
    return tuple(median_rgb)
