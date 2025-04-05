#!/usr/bin/env python3

# Typing
from typing import List

# Python
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

# PIL
from PIL import Image, ImageDraw

# Matplotlib
import matplotlib.pyplot as plt

# Src
from sba.tree_parser import TreeParser
from sba.constants import camera_intrinsic_matrix
from sba.read_poses_utils import read_poses
from sba.cylinder import Cylinder

# Utils
from sba.utils.pose import Pose
from sba.utils.cameras import PinholeCamera
from sba.utils.folder import create_folders

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

def main(args):

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.DEBUG)
    logging.getLogger("sba.read_poses_utils").setLevel(logging.WARNING)
    logger.setLevel(logging.DEBUG)

    # Matplotlib
    plt.ioff()

    # Camera intrinsics
    K = camera_intrinsic_matrix(args.camera_intrinsics)

    # Folders
    folder = Path(args.folder) / "run"
    folder_steps = folder / "optim_steps"
    h5_folder = Path(args.h5_folder)
    color_path = h5_folder / "color"
    semantic_path = h5_folder / "semantic"
    output_folder = folder / "visualization"
    create_folders(output_folder)

    # Read output of each optimization step
    idx = 0
    cylinders: List[List[Cylinder]] = []
    camera_poses: List[List[Pose]] = []
    while True:
        folder_step = Path(folder_steps / f"step_{idx}")
        if not folder_step.exists():
            break

        cylinder_file = folder_step / "cylinders.txt"
        camera_poses_file = folder_step / "text" / "images.txt"
        if not cylinder_file.is_file():
            break
        if not camera_poses_file.is_file():
            break

        # Read camera poses
        try:
            camera_poses.append(read_poses(camera_poses_file, "colmap_text"))
        except:
            break

        # Read cylinders
        cylinders.append(Cylinder.from_text_file(cylinder_file))
        idx = idx + 1

    assert len(cylinders) > 0 and "No optimization log files found."
    assert len(cylinders) == len(camera_poses)
    logger.info(f"Registered {len(cylinders)} optimization steps, which each" +
                f" contain {len(cylinders[0])} cylinder(s) and " +
                f"{len(camera_poses[0])} camera pose(s).")

    for idx in tqdm(range(len(cylinders)), desc="Steps"):

        assert len(cylinders[idx]) == 1 and "Only one cylinder supported."

        cylinder: Cylinder = cylinders[idx][0]
        ax = cylinder.visualize()
        ax.set_xlim(-2, -8)
        ax.set_ylim(-2, -8)
        ax.set_zlim(-1, 5)
        plt.savefig(output_folder / f"vis_optim_step_{idx}.png")
        plt.close()

        for camera_pose in tqdm(camera_poses[idx], total=len(camera_poses[idx]),
                                leave=False, desc="Cameras"):

            # Image paths
            img_name = camera_pose.name
            img_file = color_path / img_name
            semantic_file = semantic_path / f"{img_name[:-4]}_semantic.JPG"

            # Open the images
            image = Image.open(img_file)
            semantic = Image.open(semantic_file)

            # Create a separate image for the polygon with an alpha channel
            cylinder_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
            cylinder_draw = ImageDraw.Draw(cylinder_image)

            # Project circles
            camera = PinholeCamera(K, camera_pose)
            circle_1, circle_2 = cylinder.project_circles_to_points(camera)

            # Project polygon
            polygon = cylinder.project_edge_points(camera)

            # Draw the polygon on the transparent image
            vertices = [tuple(point) for point in polygon]
            outline_color = (230, 230, 0, 255)
            fill_color = (230, 230, 0, 198)
            border_width = 5
            cylinder_draw.polygon(vertices, outline=outline_color, fill=fill_color,
                                  width=border_width)

            for i, circle in enumerate([circle_1, circle_2]):
                if len(circle) == 0:
                    continue
                vertices = [tuple(point) for point in circle]

                if i == 0:
                    outline_color = (153, 153, 0, 255)
                    fill_color = (153, 153, 0, 198)
                else:
                    outline_color = (204, 204, 0, 255)
                    fill_color = (204, 204, 0, 198)

                border_width = 5
                cylinder_draw.polygon(vertices, outline=outline_color, fill=fill_color,
                                      width=border_width)

            # Overlay the polygon image onto the original image
            image_output = Image.alpha_composite(image.convert("RGBA"), cylinder_image)
            semantic_output = Image.alpha_composite(semantic.convert("RGBA"), cylinder_image)

            # Save the output images
            image_output.save(output_folder / f"color_{img_name}_optim_step_{idx}.png")
            semantic_output.save(output_folder / f"semantic_{img_name}_optim_step_{idx}.png")


if __name__ == "__main__":

    parser = TreeParser(
        description="Visualize the output of the geometric semantic bundle adjustment."
    )

    parser.add_argument("--folder", type=str, required=True, default=None,
                         help='The visualization output folder')

    parser.add_camera_intrinsics_argument()
    parser.add_h5_folder_argument()

    args = parser.parse_args()

    main(args)