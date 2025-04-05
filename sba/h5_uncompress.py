#!/usr/bin/env python3

# Python
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Union

# tree
from sba.h5_utils import h5_extract, save_img
from sba.utils.folder import create_folders
from sba.constants import semantic_img_to_rgb, depth_img_to_rgb

# Logging
from sba.utils.loggers import get_logger
import logging
logger = get_logger(__name__)

def h5_uncompress(h5_folder: Union[str, Path], max_depth: float = 60):

    # Transform to Path
    if type(h5_folder) == str:
        h5_folder = Path(h5_folder)

    # Verifying that the input folder exists
    if not h5_folder.exists():
        raise FileExistsError(f"The input folder '{h5_folder}' doesn't exist.")

    # Get all .h5 files in the folder
    h5_files = list(h5_folder.glob('*.h5'))

    if len(h5_files) == 0:
        raise ValueError(f"The input folder '{h5_folder}' doesn't contain any .h5 file.")

    # Create extra color image output folder
    color_folder = h5_folder / "color"
    depth_folder = h5_folder / "depth"
    semantic_folder = h5_folder / "semantic"
    create_folders(color_folder, depth_folder, semantic_folder)

    for h5_file in tqdm(h5_files, desc="Extract .h5 files"):

        # Extract h5 files
        color_img, depth_img, semantic_img =  h5_extract(h5_file)

        filename = h5_file.stem # ending with `.JPG`
        filename_without_ext = filename[:-4] # without extension, i.e removed `.JPG.h5`

        # Save all the contained images
        save_img(depth_img_to_rgb(depth_img, max_depth), depth_folder / (filename_without_ext + "_depth.JPG"))
        save_img(semantic_img_to_rgb(semantic_img), semantic_folder / (filename_without_ext + "_semantic.JPG"))
        save_img(color_img, color_folder / filename)

    N = len(h5_files)
    logger.info(f"Successfully uncompressed {N} .h5 files and saved:\n" +
                f" - {N} color images to {color_folder}\n" +
                f" - {N} depth images to {depth_folder}\n" +
                f" - {N} semantic images to {semantic_folder}\n"
                )

def main(args):

    # Set up logging
    logger = get_logger(__name__)
    logger.setLevel(logging.DEBUG)
    get_logger("sba").setLevel(logging.INFO)

    # Call script
    h5_uncompress(args.input_folder)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Reads and uncompress .h5 files in the same folder."
    )

    parser.add_argument("input_folder", help = "Input folder containing different .h5 files.")

    parser.add_argument("--max_depth", type=float, required=False, default=60,
                        help = "Maximum depth value for the depth image visualization.")

    args = parser.parse_args()
    main(args)