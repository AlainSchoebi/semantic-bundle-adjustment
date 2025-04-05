# Typing
from typing import List

# Python
import os.path

# Utils
from sba.utils.pose import Pose

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)


def check_camera_poses_and_h5_files_order(camera_poses: List[Pose], h5_files: List[str]) -> bool:

    # Verify that the number of camera poses equals the number of .h5 files
    if not len(h5_files) == len(camera_poses):
        logger.error(f"There are {len(camera_poses)} camera poses and {len(h5_files)} which are not equal.")
        return False
    if len(camera_poses) == 0:
        logger.error(f"There are no camera poses nor .h5 files.")
        return False

    # Check that the 'name' attribute exists
    if not hasattr(camera_poses[0], "name"):
        logger.error(f"The camera poses dont seem to have a 'name' attribute which is required to match the .h5 files.")
        return False

    # Check if they are matching
    for camera_pose, h5_file in zip(camera_poses, h5_files):

        # Get the exact image name corresponding to the .h5 file path
        filename = os.path.basename(h5_file)
        if filename[-3:] == ".h5": filename = filename[:-3]

        # Find corresponding camera pose
        if camera_pose.name == filename:
            logger.debug(f"Found corresponding camera poses for image '{filename}'.")
        else:
            logger.error(f"Could not find corresponding camera poses for image '{filename}'. "
                       + f"The corresponding camera poses has name '{camera_pose}'.")
            return False

    return True


def match_camera_poses_and_h5_files_order(camera_poses: List[Pose], h5_files: List[str]) -> None:
    """
    Modifies in place the lists!
    """

    # Verify that the number of camera poses equals the number of .h5 files
    if not len(h5_files) == len(camera_poses):
        raise ValueError(f"There are {len(camera_poses)} camera poses and {len(h5_files)} .h5 files which are not equal.")

    if len(camera_poses) == 0:
        raise ValueError(f"There are no camera poses nor .h5 files.")

    if not hasattr(camera_poses[0], "name"):
        raise ValueError(f"The camera poses dont seem to have a 'name' attribute which is required to match the .h5 files.")

    def parse(s: str):
        return s

    # Sort
    camera_poses.sort(key = lambda pose: parse(pose.name))
    h5_files.sort(key = lambda path: parse(os.path.basename(path)))

    if not check_camera_poses_and_h5_files_order(camera_poses, h5_files):
        raise ValueError(f"Matching camera poses and h5 files order failed.")