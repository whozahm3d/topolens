"""Image statistics calculations for ink-coverage analysis."""

from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage


def compute_ink_fraction(image_path: Path | str, white_threshold: int = 250) -> float:
    """Compute the fraction of non-white pixels in the image.

    Non-white pixels represent "ink" coverage (nodes, edges, text, boundaries).
    """
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB"))
    
    # Non-white: any RGB channel is less than white_threshold
    non_white = (arr < white_threshold).any(axis=-1)
    return float(np.mean(non_white))


def compute_component_stats(image_path: Path | str, white_threshold: int = 250) -> tuple[int, float]:
    """Label connected components on the non-white pixel mask and return stats.

    Returns:
        (num_components, mean_component_pixel_area)

    Note:
        This is most informative for sparse or disconnected graphs where individual
        components correspond directly to node blobs. In well-connected graphs,
        rendered edges typically connect and merge everything into a single large
        connected component (meaning num_components collapses to 1, and the mean
        area tracks the total ink coverage). This collapsing behavior is a reportable
        structural property of the rendering method, not a bug.
    """
    with Image.open(image_path) as img:
        arr = np.array(img.convert("RGB"))
    
    non_white = (arr < white_threshold).any(axis=-1)
    
    # Label components using scipy.ndimage.label
    labeled_array, num_features = scipy.ndimage.label(non_white)
    
    if num_features == 0:
        return 0, 0.0
    
    # Count sizes. bincount returns counts for each label value (0 is background).
    # We skip index 0.
    sizes = np.bincount(labeled_array.ravel())
    component_sizes = sizes[1:]
    
    mean_area = float(component_sizes.mean())
    return int(num_features), mean_area
