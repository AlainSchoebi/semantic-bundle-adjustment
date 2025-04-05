#!/usr/bin/env python3

# Typing
from typing import List

# Python
import numpy as np

# Src
from sba.tree_parser import TreeParser
from sba.read_poses_utils import read_poses

# Utils
from sba.utils.pose import Pose

# Logging
from sba.utils.loggers import get_logger
import logging
logger = get_logger(__name__)


def evaluate_camera_poses(camera_poses_1: List[Pose],
                          camera_poses_2: List[Pose]):

    # Checks
    if not len(camera_poses_1) == len(camera_poses_2):
        logger.error("The two camera poses list do not have the same number of poses.")
        raise ValueError("The two camera poses list do not have the same number of poses.")

    for pose in camera_poses_1 + camera_poses_2:
        if not hasattr(pose, "name"):
            logger.error("Some camera poses do not have a 'name' attribute.")
            raise ValueError("Some camera poses do not have a 'name' attribute.")

    # Error lists
    distance_errors = []
    angular_errors = []

   # Loop through every image in the model
    for pose_1 in camera_poses_1:

        found_pose = False
        for pose_2 in camera_poses_2:

            # Find corresponding image
            if pose_1.name == pose_2.name:
                found_pose = True

                # Compute error
                dt, dr = Pose.error(pose_1, pose_2, degrees=True)
                distance_errors.append(dt)
                angular_errors.append(dr)

                break

        if not found_pose:
            logger.error(f"No matching pose found for '{pose_1.name}'.")
            raise ValueError(f"No matching pose found for '{pose_1.name}'.")

    # Print
    logger.debug(f"Translation errors: {distance_errors}")
    logger.debug(f"Angular errors: {distance_errors}")

    logger.info(f"**Translation error: {np.mean(distance_errors)} (mean), {np.median(distance_errors)} (median) [m].**")
    logger.info(f"**Angular error: {np.mean(angular_errors)} (mean), {np.median(angular_errors)} (median) [°].**")

    return


def main(args):

    # Set up logging
    logger.setLevel(logging.DEBUG)
    get_logger("sba").setLevel(logging.INFO)

    # Camera poses
    camera_poses_1 = read_poses(args.camera_poses_path_1, args.camera_poses_format_1)
    camera_poses_2 = read_poses(args.camera_poses_path_2, args.camera_poses_format_2)

    # Evaluate
    evaluate_camera_poses(camera_poses_1, camera_poses_2)


if __name__ == "__main__":

    parser = TreeParser(
        description="Compares two aligned set of camera poses."
    )

    parser.add_argument(
        "--camera_poses_path_1", type=str, required=False, default=None,
        help = "Input file or folder containing the camera poses."
    )

    parser.add_argument(
        "--camera_poses_path_2", type=str, required=False, default=None,
        help = "Input file or folder containing the camera poses."
    )

    parser.add_argument(
            "--camera_poses_format_1", type=str, required=False, default=None,
            choices=["vulkan_text", "colmap_text", "colmap_model"],
            help = "'vulkan_text': provide a .txt file generaed with Vulkan. " +
                   "'colmap_text': provide  a .txt file exported form COLMAP. " +
                   "'colmap_model': provide a folder model folder form COLMAP."
    )

    parser.add_argument(
            "--camera_poses_format_2", type=str, required=False, default=None,
            choices=["vulkan_text", "colmap_text", "colmap_model"],
            help = "'vulkan_text': provide a .txt file generaed with Vulkan. " +
                   "'colmap_text': provide  a .txt file exported form COLMAP. " +
                   "'colmap_model': provide a folder model folder form COLMAP."
    )

    args = parser.parse_args()

    main(args)