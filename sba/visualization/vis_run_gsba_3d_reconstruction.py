#!/usr/bin/env python3

# Python
import logging
import os.path
from pathlib import Path
from tqdm import tqdm

# Src
from sba.tree_parser import TreeParser
from sba.h5_utils import get_all_h5_files_in_folder
from sba.constants import camera_intrinsic_matrix
from sba.read_poses_utils import read_poses
from sba.reconstruction_3d import reconstruct_3d
from sba.cylinder import Cylinder

# Utils
from sba.utils.loggers import get_logger
from sba.utils.folder import create_folder

def main(args):

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.WARNING)
    logger = get_logger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    output_folder = Path(args.output_folder)
    create_folder(output_folder)

    folder = Path(args.folder)
    if not folder.exists():
        logger.error(f"The provided folder '{folder}' does not exist.")
        return

    optim_steps = folder / "optim_steps"
    if not optim_steps.exists():
        logger.error(f"The provided folder '{folder}' does have a 'optim_steps' subfolder.")
        return

    # Parameters
    h5_files = get_all_h5_files_in_folder(args.h5_folder)
    K = camera_intrinsic_matrix(args.camera_intrinsics)

    # Find folders
    step_folders = []
    i = 0
    while (optim_steps / f"step_{i}").exists():
        step_folders.append(optim_steps / f"step_{i}")
        # Next step
        i = i + 1

    logger.info(f"Found {len(step_folders)} optimization step(s).")

    # Loop through each step
    for i, step_folder in tqdm(enumerate(step_folders), total=len(step_folders), desc="Steps"):

        # Read camera poses
        camera_poses = read_poses(step_folder, "colmap_model")

        # Read cylinders
        cylinders = Cylinder.from_text_file(step_folder / "cylinders.txt")

        # Output bag
        output_bag=output_folder / f"step_{i}.bag",

        # 3D reconstruction
        reconstruct_3d(
            camera_poses=camera_poses,
            h5_files=h5_files,
            K=K,
            output_bag=output_bag,
            SAVE_IMAGES=False,
            cylinders=cylinders
        )

        logger.info(f"Successfully saved out bag for step f{i} to '{output_bag}'.")

if __name__ == "__main__":

    parser = TreeParser(
        description="Visualize the output of the geometric semantic bundle adjustment."
    )

    parser.add_argument("--folder", type=str, required=True, default=None,
                         help='The visualization output folder')

    parser.add_h5_folder_argument()

    parser.add_camera_intrinsics_argument()

    parser.add_argument("-o", "--output_folder", type=str, required=False, default="out/last_gsba_reconstructions",
                        help="Output folder containing the different reconstructions.")

    args = parser.parse_args()

    main(args)