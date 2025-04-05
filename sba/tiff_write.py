#!/usr/bin/env python3

# Python
import numpy as np
import argparse
import os.path
from tqdm import tqdm
from pathlib import Path
from typing import Union

# Tiff library
import tifffile

# src
from sba.h5_utils import h5_extract
from sba.utils.folder import create_folders
from sba.constants import semantic_reassign

# Logging
from sba.utils.loggers import get_logger
import logging
logger = get_logger(__name__)

def tiff_write(h5_folder: Union[str, Path]):

    h5_folder = Path(h5_folder)

    # Verifying the paths
    if not h5_folder.exists():
        raise FileExistsError(
            f"The provided h5_folder not exist, i.e. '{str(h5_folder)}' doesn't exist."
        )

    # Get all .h5 files in the folder
    h5_files = list(h5_folder.glob('*.h5'))

    # Create extra color image output folder
    color_folder = h5_folder / "color_tiff"
    depth_folder = h5_folder / "depth_tiff"
    semantic_folder = h5_folder / "semantic_tiff"
    create_folders(str(color_folder), str(depth_folder), str(semantic_folder))

    for h5_file in tqdm(h5_files, desc="Writing .tiff files"):

        # Read the .h5 file
        color_img, depth_img, semantic_img = h5_extract(str(h5_file))

        # Get the file names
        filename = os.path.basename(str(h5_file))[:-3] # ending with `.JPG`
        filename_without_ext = filename[:-4] # without `.JPG.h5`

        # Write the color .tiff file
        color_tiff = color_img
        color_path = color_folder / (filename_without_ext + "_color.tiff")
        tifffile.imwrite(color_path, color_tiff)

        # Write the depth .tiff file
        depth_tiff = depth_img.astype(np.float32)
        depth_path = depth_folder / (filename_without_ext + "_depth.tiff")
        tifffile.imwrite(depth_path, depth_tiff, dtype=np.float32)

        # Write the semantic .tiff file
        semantic_tiff = semantic_reassign(semantic_img).astype(np.float32)
        semantic_path = semantic_folder /\
              (filename_without_ext + "_semantic.tiff")
        tifffile.imwrite(semantic_path, semantic_tiff, dtype=np.float32)

    N = len(h5_files)
    logger.info(f"Successfully saved:\n" +
                f" - {N} color uint8? .tiff images under {color_folder}\n" +
                f" - {N} depth float32 .tiff images under {depth_folder}\n" +
                f" - {N} semantic float32 .tiff images under {semantic_folder}\n"
                )

def main(args):

    # Set up logging
    logger = get_logger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)
    get_logger("sba").setLevel(logging.INFO)

    folder = Path(args.input_folder)
    h5_folder = folder / "output"

    # Verifying that the input folder exists
    if not folder.exists():
        raise FileExistsError(
            f"The input folder '{str(folder)}' doesn't exist."
        )

    tiff_write(h5_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reads .h5 files and write tiff files."
    )

    parser.add_argument("input_folder", help = "Result folder of vrg_crop_gen.")

    args = parser.parse_args()
    main(args)