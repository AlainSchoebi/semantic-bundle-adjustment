#!/usr/bin/env python3

# Typing
from typing import List, Union

# Python
import logging
from pathlib import Path

# Src
from sba.tree_parser import TreeParser
from sba.read_poses_utils import read_poses

# Utils
from sba.utils.pose import Pose

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)


def geo_registration_file(camera_poses: List[Pose], output_file: Union[str, Path]) -> None:
    """
    Write the geo-registration text file required by COLMAP for a geo-registration.

    Inputs
    - camera_poses: list of camera poses
    - output_file:  the output text file that will contain the camera poses locations

    Outputs
    - a text file is created
    """
    for camera_pose in camera_poses:
        if not hasattr(camera_pose, "name"):
            logger.error("The camera poses must have a 'name' attribute")
            raise ValueError("The camera poses must have a 'name' attribute")

    output_file = Path(output_file)
    if not output_file.suffix == ".txt":
        logger.error(f"The geo-registration file must be a '.txt' file, not '{output_file.suffix}'.")
        raise ValueError(f"The geo-registration file must be a '.txt' file, not '{output_file.suffix}'.")

    with open(output_file, "w") as text_file:

        # Loop
        camera_pose: Pose
        for camera_pose in camera_poses:
            name = camera_pose.name
            t = camera_pose.t
            text_file.write(f"{name} {t[0]} {t[1]} {t[2]}\n")

    logger.info(f"Successfully wrote {len(camera_poses)} lines to the text file '{output_file}'.")


def main(args):

    # Step up logging
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.DEBUG)

    # Camera poses array
    camera_poses = read_poses(args.camera_poses_path, args.camera_poses_format)

    # Create geo-registration file
    geo_registration_file(camera_poses, args.output)


if __name__ == "__main__":

    parser = TreeParser(
        description="Writes the geo-registration text file required by the colmap geo-registration command.",
    )

    parser.add_camera_poses_arguments()

    parser.add_argument("--output", "-o", type=str, required=False, default = "out/ground_truth_geo_registration.txt",
                        help = "Output text file in the format required by the colmap geo reigstration.")

    args = parser.parse_args()

    main(args)