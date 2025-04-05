#!/usr/bin/env python3

# Typing
from typing import List, Any, Dict

# Python
import numpy as np
import logging
import pandas as pd
import os.path
from pathlib import Path
from enum import Enum

# PIL
from PIL import Image, ImageDraw

# Utils
from sba.utils.folder import create_folders

# Src
from sba.tree_parser import TreeParser

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

DRAW_LINES = True # TODO

class ReprojStatus(Enum):
    VALID_DEPTH = 10
    BAD_DEPTH = -2
    OUT_OF_BOUNDS = -1

SIDE_BY_SIDE_IMAGES = False
OTHER_IMAGES = False

BAD_DEPTH_COLOR = "red"

def visualize_sba_csv(csv_file: Path, h5_folder: Path, img_1: str, img_2: str, output_path: Path, prefix: str) -> None:

    # Access color, depth and semantic images
    color_path = h5_folder / "color"
    depth_path = h5_folder / "depth"
    semantic_path = h5_folder / "semantic"

    img_file_1 = color_path / img_1
    img_file_2 = color_path / img_2

    depth_file_1 = depth_path / f"{img_1[:-4]}_depth.JPG"
    depth_file_2 = depth_path / f"{img_2[:-4]}_depth.JPG"

    semantic_file_1 = semantic_path / f"{img_1[:-4]}_semantic.JPG"
    semantic_file_2 = semantic_path / f"{img_2[:-4]}_semantic.JPG"

    # Read the CSV file into a pandas DataFrame
    if not csv_file.is_file():
        logger.warn(f"{csv_file.name} does not exist. Skip.")
        return
    with open(csv_file, 'r') as file:
        df = pd.read_csv(file)

    types = df['Type'].values
    semantic_errors = df['SemanticError'].values
    points_1 = np.array([df['X1'].values, df['Y1'].values]).T
    points_2 = np.array([df['X2'].values, df['Y2'].values]).T
    points_3D = np.array([df['X3D'].values, df['Y3D'].values, df['Z3D'].values]).T # Not being used

    # Print infomration
    logger.info(f"\nImage {img_1} -> {img_2}:")
    logger.info(f"  - Registered {len(semantic_errors)} point correspondences, out of which:")
    logger.info(f"       - {np.sum(types == ReprojStatus.OUT_OF_BOUNDS.value):5.0f} were out of bounds")
    logger.info(f"       - {np.sum(types == ReprojStatus.BAD_DEPTH.value):5.0f} had bad depth")
    logger.info(f"       - {np.sum(types == ReprojStatus.VALID_DEPTH.value):5.0f} had valid depth")
    logger.info(f"  - Out of the {np.sum(types == ReprojStatus.VALID_DEPTH.value)} correspondences with valid depth:")
    logger.info(f"       - {np.sum(semantic_errors[types == ReprojStatus.VALID_DEPTH.value] == 0):5.0f} " +
          f"had correct semantics")
    logger.info(f"       - {np.sum(semantic_errors[types == ReprojStatus.VALID_DEPTH.value] == 1):5.0f} " +
          f"had incorrect semantics")
    logger.info("")

    # Compute semantic error
    valid_semantic_errors = semantic_errors[types == ReprojStatus.VALID_DEPTH.value]
    n_wrong_pixels = np.sum(valid_semantic_errors > 0)
    logger.info(f"{n_wrong_pixels} pixels had wrong semantic classes out of {valid_semantic_errors.size} pixels, " +
          f"i.e. {n_wrong_pixels/valid_semantic_errors.size*100:.2f}%.")

    assert len(types) == len(points_1) and len(types) == len(points_2)

    # Open the image files
    image_1 = Image.open(img_file_1)
    image_2 = Image.open(img_file_2)

    depth_1 = Image.open(depth_file_1)
    depth_2 = Image.open(depth_file_2)

    semantic_1 = Image.open(semantic_file_1)
    semantic_2 = Image.open(semantic_file_2)

    width_1, height_1 = image_1.size
    width_2, height_2 = image_2.size

    assert width_1 == width_2 and height_1 == height_2

    draws = []
    if SIDE_BY_SIDE_IMAGES:
        # Determine the size of the concatenated image
        total_width = width_1 + width_2
        max_height = height_1

        # Create a new image with the determined size
        image = Image.new('RGB', (total_width, max_height), color='white')
        image.paste(image_1, (0, 0))
        image.paste(image_2, (width_1, 0))

        depth = Image.new('RGB', (total_width, max_height), color='white')
        depth.paste(depth_1, (0, 0))
        depth.paste(depth_2, (width_1, 0))

        semantic = Image.new('RGB', (total_width, max_height), color='white')
        semantic.paste(semantic_1, (0, 0))
        semantic.paste(semantic_2, (width_1, 0))

        draw_color = ImageDraw.Draw(image)
        draw_depth = ImageDraw.Draw(depth)
        draw_semantic = ImageDraw.Draw(semantic)

        draws = [draw_color, draw_depth, draw_semantic]

    # Overlay the images
    if OTHER_IMAGES:
        draw_overlay_blank = ImageDraw.Draw(image_2_overlayed_on_1_blank)
        image_2_overlayed_on_1_blank = Image.new('RGB', (width_1, height_1), color='white')
        image_2_overlayed_on_1_all = Image.new('RGB', (width_1, height_1), color='white')
        draw_overlay_all = ImageDraw.Draw(image_2_overlayed_on_1_all)
        semantic_1_adapted_colors = semantic_1.convert("L").convert("RGB")
        draw_overlay = ImageDraw.Draw(image_2_overlayed_on_1)

    image_2_overlayed_on_1 = image_1.convert("L").convert("RGB")

    semantic_2_overlayed_on_1 = semantic_1.convert("L").convert("RGB")
    draw_semantic_overlay = ImageDraw.Draw(semantic_2_overlayed_on_1)

    overlay_draws = []
    if OTHER_IMAGES:
        overlay_draws = [draw_overlay, draw_overlay_blank]

    semantic_overlay_draws = [draw_semantic_overlay]
    for i, (reproj_status_int, semantic_error, point_1, point_2) in \
          enumerate(zip(types, semantic_errors, points_1, points_2)):

        center_1 = (point_1[0], point_1[1])
        center_2 = (point_2[0] + width_1, point_2[1])

        status = ReprojStatus(reproj_status_int)

        x1, y1 = point_1[0], point_1[1]
        r = 3
        if not status == ReprojStatus.OUT_OF_BOUNDS:
            color = image_2.getpixel((point_2[0], point_2[1]))

            semantic_color = semantic_2.getpixel((point_2[0], point_2[1]))
            if status == ReprojStatus.BAD_DEPTH:
                semantic_color = BAD_DEPTH_COLOR
            for d in semantic_overlay_draws:
                d.ellipse([x1-r, y1-r, x1+r, y1+r], outline=None, fill=semantic_color)
        else:
            color = 'blue'
        if status == ReprojStatus.VALID_DEPTH:
            for d in overlay_draws:
                d.ellipse([x1-r, y1-r, x1+r, y1+r], outline=None, fill=color)

        if OTHER_IMAGES:
            if status == ReprojStatus.VALID_DEPTH or status == ReprojStatus.BAD_DEPTH:
                draw_overlay_all.ellipse([x1-r, y1-r, x1+r, y1+r], outline=None, fill=color)
            else:
                draw_overlay_all.ellipse([x1-r, y1-r, x1+r, y1+r], outline=None, fill='black')


        if i%400 != 0: # TODO
            continue

        if status == ReprojStatus.OUT_OF_BOUNDS:
            radius = 5

            for d in draws:
                d.ellipse((center_1[0] - radius, center_1[1] - radius, center_1[0] + radius, center_1[1] + radius),
                           fill='white')

        elif status == ReprojStatus.BAD_DEPTH:
            radius = 8
            for d in draws:
                d.ellipse((center_1[0] - radius, center_1[1] - radius, center_1[0] + radius, center_1[1] + radius),
                           fill='blue')
                d.ellipse((center_2[0] - radius, center_2[1] - radius, center_2[0] + radius, center_2[1] + radius),
                           fill='blue')
                if DRAW_LINES:
                   d.line([center_1, center_2], fill=BAD_DEPTH_COLOR, width=2)

        elif status == ReprojStatus.VALID_DEPTH:
            radius = 10
            color = 'green' if semantic_error == 0 else 'red'
            for d in draws:
                d.ellipse((center_1[0] - radius, center_1[1] - radius, center_1[0] + radius, center_1[1] + radius),
                           fill=color)
                d.ellipse((center_2[0] - radius, center_2[1] - radius, center_2[0] + radius, center_2[1] + radius),
                           fill=color),
                if DRAW_LINES:
                    d.line([center_1, center_2], fill=color, width=3)

    if SIDE_BY_SIDE_IMAGES:
        image.save(output_path / (prefix + f"_{img_1}_to_{img_2}.png"))
        depth.save(output_path / (prefix + f"_{img_1}_to_{img_2}_depth.png"))
        semantic.save(output_path / (prefix + f"_{img_1}_to_{img_2}_semantic.png"))

    if OTHER_IMAGES:
        image_2_overlayed_on_1_all.save(output_path / (prefix + f"_{img_1}_to_{img_2}_overlay_all.png"))
        image_2_overlayed_on_1_blank.save(output_path / (prefix + f"_{img_1}_to_{img_2}_overlay_blank.png"))
        image_2_overlayed_on_1.save(output_path / (prefix + f"_{img_1}_to_{img_2}_overlay.png"))
        semantic_1_adapted_colors.save(output_path / (prefix + f"_{img_1}_to_{img_2}_semantic_1.png"))

    semantic_2_overlayed_on_1.save(output_path / (prefix + f"_{img_1}_to_{img_2}_semantic_overlay.png"))

def main(args):

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.DEBUG)
    logging.getLogger(__name__).setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # Folders
    folder = Path(args.folder)
    folder_steps = folder / "run" / "optim_steps"
    h5_folder = Path(args.h5_folder)
    output_folder = folder / "run" / "visualization"
    create_folders(output_folder)

    # Read output of each optimization step
    idx = 0
    csvs: List[List[Dict[str, Any]]] = []
    while True:
        folder_step = Path(folder_steps / f"step_{idx}")
        if not folder_step.exists():
            break

        csv_files_step = list(folder_step.glob('*.csv'))

        csvs.append([])
        for k, csv_file in enumerate(csv_files_step):
            csvs[idx].append([])
            names = csv_file.name.replace(".csv", "").replace("vis_", "").split("_to_")
            assert len(names) == 2
            csvs[idx][k] = {"csv": csv_file, "name_1": names[0], "name_2": names[1], "step": idx}

        idx = idx + 1

    assert len(csvs) > 0 and "No optimization step files found."

    for csvs_step in csvs:
        assert len(csvs_step) == len(csvs[0]) and "Number of .csv files in each folder must be equal."

    logger.info(f"Found {len(csvs)} optimization steps that each contain {len(csvs[0])} csv files.")

    for csvs_step in csvs:
        for csv in csvs_step:
            if csv["name_1"] == "IMG4.JPG" and csv["name_2"] == "IMG0.JPG":
                visualize_sba_csv(csv["csv"], h5_folder, csv["name_1"], csv["name_2"], output_folder, f"step_{csv['step']}")

if __name__ == "__main__":

    parser = TreeParser(
        description="Visualize the output of the semantic bundle adjustment."
    )
    parser.add_argument("--folder", type=str, required=True, default=None,
                         help='The visualization output folder')
    parser.add_h5_folder_argument()

    args = parser.parse_args()

    main(args)