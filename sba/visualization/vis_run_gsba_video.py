#!/usr/bin/env python3

# Python
import numpy as np
import logging
from pathlib import Path

# Src
from sba.tree_parser import TreeParser

# Utils
from sba.utils.folder import create_folders
from sba.utils.video import generate_video, generate_4_tile_video

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)


def main(args):

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # Folders
    folder = Path(args.folder) / "run"
    image_folder = folder / "visualization"

    # Folder checks
    if not folder.exists():
        logger.error(f"The provided folder {folder} does not exist.")
        raise ValueError(f"The provided folder {folder} does not exist.")

    if not image_folder.exists():
        logger.error(f"The provided folder {folder} does not have a '/visualization' folder.")
        raise ValueError(f"The provided folder {folder} does not have a '/visualization' folder.")

    # Output videos
    video_folder = folder / "videos"
    create_folders(video_folder)

    # Get all PNG images
    all_images = list(image_folder.glob('*.png'))
    #
    color_images = [img for img in all_images if "color_" in img.name]
    semantic_images = [img for img in all_images if "semantic_" in img.name]
    cylinder_images = [img for img in all_images if "vis_" in img.name]

    # Extract the substring between 'color_' and '_optim_'
    def extract(name: str) -> str:
        c = "color_"
        s = "semantic_"
        if name[0:len(c)] == c:
            start_idx = len(c)
        elif name[0:len(s)] == s:
            start_idx = len(s)
        else:
            raise ValueError("Incorrect image name.")

        end_idx = name.find("_optim_")
        if end_idx == -1:
            raise ValueError("Incorrect image name.")

        return name[start_idx:end_idx]

    # Get image names
    all_names = [extract(img.name) for img in color_images]
    names = np.unique(all_names)
    all_names = [extract(img.name) for img in semantic_images]
    assert sorted(np.unique(all_names)) == sorted(names)

    # Sort images per image name
    get_step_number = lambda img: int(img.name.split("_optim_step_")[1].replace(".png", ""))

    color_images_corners = []
    semantic_images_corners = []
    for name in names:
        color_images_per_name = [img for img in color_images if name in img.name]
        color_images_per_name.sort(key = get_step_number)

        generate_video(color_images_per_name, video_folder / f"color_video_{name}.mp4")

        semantic_images_per_name = [img for img in semantic_images if name in img.name]
        semantic_images_per_name.sort(key = get_step_number)

        generate_video(semantic_images_per_name, video_folder / f"semantic_video_{name}.mp4")

        if len(color_images_corners) < 4:
            color_images_corners.append(color_images_per_name)
            semantic_images_corners.append(semantic_images_per_name)

    if len(color_images_corners) == 4:
        generate_4_tile_video(color_images_corners, video_folder / f"color_video_4_images.mp4")
        generate_4_tile_video(semantic_images_corners, video_folder / f"semantic_video_4_images.mp4")

if __name__ == "__main__":

    parser = TreeParser(
        description="Generate a video from the visualization step images."
    )

    parser.add_argument("--folder", type=str, required=True, default=None,
                         help="The GSBA visualization output folder. It must contain a '/visualization' folder.")

    args = parser.parse_args()

    main(args)