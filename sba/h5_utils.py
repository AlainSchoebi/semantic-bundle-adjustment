# Python
import numpy as np
import h5py
from pathlib import Path

# Typing
from typing import Tuple, List, Union
from numpy.typing import NDArray

# Image
import PIL.Image
import matplotlib


def h5_extract(file: Union[str, Path]) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Returns the extracted color image, depth image and semantic image from a .h5 file given as input.
    """

    # Get h5 object
    f = h5py.File(str(file), 'r')

    # Check for the keys
    if "depth_data" not in list(f.keys()):
        raise ValueError(f"The .h5 file '{file}' doesn't contain 'depth_data'.")
    if "rgbs_data" not in list(f.keys()):
        raise ValueError(f"The .h5 file '{file}' doesn't contain 'rgbs_data'.")

    # Extract all the different images
    depth_img = np.array(f["depth_data"][0, :, :], dtype=np.float32) # float
    rgbs_data = np.transpose(f["rgbs_data"], (1,2,0)).astype(np.uint8)
    semantic_img = rgbs_data[:, :, 3]
    color_img = rgbs_data[: ,:, [2,1,0]]

    return color_img, depth_img, semantic_img

def h5_write(file: Union[str, Path], color: NDArray, depth: NDArray, semantic: NDArray) -> None:
    """
    Create a new .h5 file containing two fields:
    - rgbs_data: 4 channel image, 3 channels for the color image and the last channel for the semantics
    - depth_data: 1 channel image containing the depth values
    """

    with h5py.File(str(file), 'w') as f:
        # Create datasets for color_data, depth_data, and semantic_data
        color = color[:, :, [2, 1, 0]]
        rgbs_data = np.concatenate((color, semantic[..., None]), axis = -1) # (H, W, 4), np.uint8
        rgbs_data = np.transpose(rgbs_data, (2, 0, 1))
        f.create_dataset('rgbs_data', data=rgbs_data)
        f.create_dataset('depth_data', data=depth[None, :, :])


def get_all_h5_files_in_folder(folder: Union[Path, str]) -> List[Path]:

    h5_folder = Path(folder)

    # Verify that the input folder containing the .h5 files exists
    if not h5_folder.exists():
        raise FileExistsError(f"The input folder containing the .h5 files '{folder}' doesn't exist.")

    h5_files = list(h5_folder.glob('*.h5'))
    h5_files.sort(key = lambda h: str(h))

    return h5_files

def save_img(img: NDArray, path: str, colormap: matplotlib.colors.Colormap = matplotlib.cm.get_cmap('viridis')) -> None:

    # Verify image type
    if img.dtype != np.uint8:
        if img.dtype == np.float32:
            if np.max(img) > 1:
                raise ValueError(f"Image of type 'np.float32' has a maximum value of '{np.max(img)}'.")
            img = (img * 256).astype(np.uint8)

    # Grayscale
    if img.ndim == 2:
        rgb_img = (colormap(img)[:, :, :3] * 255).astype(np.uint8)
        pil_img = PIL.Image.fromarray(rgb_img, mode = "RGB")
    # RGB
    elif img.ndim == 3 and img.shape[-1] == 3:
        pil_img = PIL.Image.fromarray(img, mode = "RGB")
    # RGBA
    elif img.ndim == 3 and img.shape[-1] == 4:
        pil_img = PIL.Image.fromarray(img, mode = "RGBA")

    # Save the image to the given path
    pil_img.save(path)
    return