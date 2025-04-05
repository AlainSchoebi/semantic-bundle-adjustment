# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import List, Tuple, Type

# Matplotlib
import matplotlib

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

# Semantic error type
SemanticError = Type[Tuple[Tuple[str, str], Tuple[float, float]]]

# Define the semantic colors
semantic_colors_rgba = {
                   0: np.array([1, 1, 1, 1], dtype=float), # undefined, sky (white)
                   50: np.array([82/255, 34/255, 2/255, 1], dtype=float), # soil
                   100: np.array([0, 0.3, 0, 1], dtype=float), # grass
                   125: np.array([0, 0, 1, 1], dtype=float),   # fence
                   250: np.array([19/255, 138/255, 15/255, 1], dtype=float), # tree
                   251: np.array([24/255, 161/255, 154/255, 1], dtype=float), # canopy
                   }

semantic_colors_rgb = {
                   0: semantic_colors_rgba[0][:3],
                   50: semantic_colors_rgba[50][:3],
                   100: semantic_colors_rgba[100][:3],
                   125: semantic_colors_rgba[125][:3],
                   250: semantic_colors_rgba[250][:3],
                   251: semantic_colors_rgba[251][:3],
                   }

def semantic_img_to_rgb(semantic_img: NDArray, with_reassign: bool = True) -> NDArray:
    assert semantic_img.ndim == 2
    if with_reassign: semantic_img = semantic_reassign(semantic_img)
    semantic_img_rgb = np.zeros((*semantic_img.shape, 3), dtype=np.uint8)
    for semantic_class in list(np.unique(semantic_img)):

            # Define the unknown semantic class color
            color = np.array([100, 100, 100])

            # Access the semantic class color if existing
            if semantic_class in semantic_colors_rgb.keys():
                color = semantic_colors_rgb[semantic_class]*255

            # Set the color
            color = color.astype(np.uint8)
            semantic_img_rgb[semantic_img == semantic_class] = color

    return semantic_img_rgb

def depth_img_to_rgb(depth_img: NDArray, max_depth: float = 200) -> NDArray:
    assert depth_img.ndim == 2
    colormap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap('viridis')
    if np.max(depth_img) > max_depth:
        logger.warning(f"The provided depth image has a maximum depth {np.max(depth_img)} which is higher than " +
                       f"the max_depth of {max_depth}. -> Clipping values.")
        depth_img = np.clip(depth_img, 0, max_depth)

    depth_img_rgb = (colormap(depth_img / max_depth)[:, :, :3] * 255).astype(np.uint8)
    return depth_img_rgb

def semantic_reassign(semantic_classes: NDArray) -> NDArray:
    output_classes = semantic_classes.copy()
    output_classes[semantic_classes == 100] = 50 # grass -> soil
    return output_classes

def camera_intrinsic_matrix(intrinsics: List[float]) -> NDArray:
    if not len(intrinsics) == 4:
        raise ValueError("The camera instrinsics must consist of 4 numbers in the following order: fx, fy, cx, cy.")
    fx, fy, cx, cy = intrinsics
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])