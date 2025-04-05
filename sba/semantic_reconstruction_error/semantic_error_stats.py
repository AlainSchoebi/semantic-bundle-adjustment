#!/usr/bin/env python3

# Numpy
import numpy as np

# Python
from typing import List, Tuple
import logging
from tqdm import tqdm
from pathlib import Path
import json

# Src
from sba.constants import camera_intrinsic_matrix
from sba.h5_utils import get_all_h5_files_in_folder
from sba.read_poses_utils import read_poses, save_poses_to_new_bag
from sba.match_poses_with_h5 import match_camera_poses_and_h5_files_order
from sba.semantic_reconstruction_error.error import compute_semantic_reconstruction_error_from_indices_pairs
from sba.tree_parser import TreeParser

# Utils
from sba.utils.pose import Pose
from sba.utils.folder import create_folder

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

# Parameters to chose
DEPTH_ERROR_THRESHOLD = 0.5
DISTANCE_ERROR_THRESHOLD = 0.5
N_SAMPLES = 30
ERROR_TYPE = "2D"  # "2D" or "3D"
POSITION_NOISES = [0, 0.2, 0.4, 0.6, 0.8, 1]  # std dev in [rad]
OUTPUT_FOLDER = Path(f"out/last_semantic_error_stats_{ERROR_TYPE}_{N_SAMPLES}_samples")
OUTPUT_ERROR_FILE = OUTPUT_FOLDER / f"errors_{ERROR_TYPE}_{N_SAMPLES}_samples.json"

def generate_last_noisy_camera_poses(camera_poses: List[Pose],
                                     position_noise: float,
                                     rotation_noise: float):
    # Reset the noisy camera poses
    noisy_camera_poses = []

    # Copy the camera poses
    camera_pose: Pose
    for camera_pose in camera_poses:
        noisy_camera_pose = camera_pose.copy()
        noisy_camera_pose.name = camera_pose.name
        noisy_camera_poses.append(noisy_camera_pose)

    # Add noise to the last camera pose
    noisy_t = camera_poses[-1].t + np.random.normal(0, position_noise, 3)
    # only add noise to the angle of the rotation ? TODO
    angle, axis = camera_poses[-1].rotation_angle_and_axis()
    noisy_angle = angle + np.random.normal(0, rotation_noise)

    # Create and replace the last noisy camera pose
    noisy_camera_poses[-1] = Pose.from_rotation_angle_and_axis(noisy_angle,
                                                               axis,
                                                               t=noisy_t)
    noisy_camera_poses[-1].name = camera_poses[-1].name

    return noisy_camera_poses


def main(args):
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)
    get_logger("sba").setLevel(logging.DEBUG)

    # Numpy random seed
    np.random.seed(1)

    # Camera poses array
    camera_poses = read_poses(args.camera_poses_path, args.camera_poses_format)
    # Get all .h5 files contained in the h5 folder (in sorted order)
    h5_files = get_all_h5_files_in_folder(args.h5_folder)
    # Check if the order has matching image names
    match_camera_poses_and_h5_files_order(camera_poses, h5_files)

    # Take a subset of the images if asked
    if args.subset is not None:
        camera_poses = camera_poses[:args.subset]
        h5_files = h5_files[:args.subset]
        logger.info(
            f"Only using a subset of {args.subset} images and camera poses.")

    # Camera intrinsics
    K = camera_intrinsic_matrix(args.camera_intrinsics)

    create_folder(OUTPUT_FOLDER)
    errors = dict()

    # Loop values
    for position_noise in POSITION_NOISES:
        all_noisy_poses = []

        errors[position_noise] = []

        for i in tqdm(range(1 if position_noise == 0 else N_SAMPLES),
                      desc="Samples"):
            noisy_camera_poses = generate_last_noisy_camera_poses(camera_poses, position_noise, 0)
            all_noisy_poses.append(noisy_camera_poses[-1])

            indices: List[Tuple[int]] = []
            last_idx = len(camera_poses) - 1
            for i in range(len(camera_poses) - 1):
                indices.append((i, last_idx))
                indices.append((last_idx, i))

            # n_error, n_pixels = np.random.random(), np.random.random()

            n_error, n_pixels = compute_semantic_reconstruction_error_from_indices_pairs(
                K, noisy_camera_poses, h5_files, indices=indices, error_type=ERROR_TYPE,
                depth_error_threshold=DEPTH_ERROR_THRESHOLD,
                distance_error_threshold=DISTANCE_ERROR_THRESHOLD)

            # Error could be 'None' if no error could be computed.
            logger.debug(f"There are in total {n_error} error pixels out of {n_pixels} pixels.")

            errors[position_noise].append([n_error, n_pixels])

        # Save to a new ROS bag for visualization
        save_poses_to_new_bag(OUTPUT_FOLDER / f"noisy_{position_noise}.bag",
            all_noisy_poses, "/noisy_poses", camera_poses, "/original_poses")

    # Save the error to a json file
    with open(OUTPUT_ERROR_FILE, 'w') as json_file:
        json.dump(errors, json_file, indent=2)

    logger.info(f"Successfully saved the statistics to '{OUTPUT_ERROR_FILE}'.")

if __name__ == "__main__":
    parser = TreeParser(description="Semantic error statistics.")

    parser.add_workspace_and_camera_h5_folder_arguments()
    parser.add_camera_intrinsics_argument()

    parser.add_argument("--subset", type=int, required=False, default=None,
                         help= "Only take a subset of the all the images and camera poses. Default is use all images",
    )

    args = parser.parse_args()

    main(args)