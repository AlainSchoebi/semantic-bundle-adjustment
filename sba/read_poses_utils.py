# Typing
from typing import List, Union
from numpy.typing import NDArray

# Python
import numpy as np
from enum import Enum
import pathlib
from pathlib import Path

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

# COLMAP
try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False

# ROS
try:
    import rosbag
    from geometry_msgs.msg import PoseArray
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Utils
from sba.utils.pose import Pose

class PosesFormat(Enum):
    VULKAN_TEXT = 0
    COLMAP_TEXT = 1
    COLMAP_MODEL = 2

def poses_format_from_string(format: str) -> PosesFormat:
    format == format.lower().strip()
    if format == "vulkan" or format == "vulkan_text":
        return PosesFormat.VULKAN_TEXT

    if format == "colmap_text":
        return PosesFormat.COLMAP_TEXT

    if format == "colmap_model":
        return PosesFormat.COLMAP_MODEL

    logger.error(f"Pose format '{format}' is not supported.")
    raise NotImplementedError(f"Pose format '{format}' is not supported.")

def read_poses(path: [str, Path], format: Union[str, PosesFormat]) -> List[Pose]:
    """
    Returns all the camera poses contained in the provided path.

    Note: the poses describe the transformation from camera to world, that is t = (tx, ty, tz) are the camera
          coordinates in the world frame.
    """
    if type(format) == str:
        format = poses_format_from_string(format)

    if format == PosesFormat.VULKAN_TEXT:
        return read_poses_from_vulkan_text_file(path)

    if format == PosesFormat.COLMAP_TEXT:
        return read_poses_from_colmap_text_file(path)

    if format == PosesFormat.COLMAP_MODEL:
        if not COLMAP_AVAILABLE:
            logger.error("COLMAP is not available. Cannot read poses from COLMAP model.")
            raise NotImplementedError("COLMAP is not available. Cannot read poses from COLMAP model.")
        return read_poses_from_colmap_model(path)

    logger.error(f"Reading the poses for the poses format '{format}' is not supported.")
    raise NotImplementedError(f"Reading the poses for the poses format '{format}' is not supported.")


def read_poses_from_vulkan_text_file(file: Union[str, Path]) -> List[Pose]:

    # Convert to Path
    file = Path(file)

    # Input file checks
    if not file.is_file():
        logger.error(f"The input file '{str(file)}' doesn't exists or is not a file.")
        raise FileExistsError(f"The input file '{str(file)}' doesn't exists or is not a file.")
    if not file.suffix == ".txt":
        logger.error(f"The input file '{file}' is not a .txt file.")
        raise TypeError(f"The input file '{file}' is not a .txt file.")

    # Camera poses array
    poses: List[Pose] = []

    # Open the text file
    with open(file) as f:

        while True:

            # Read line
            line = f.readline()
            if not line: # End of the file
                break
            if line[0] == '#': # Skip commented lines
                continue

            if line[:9] == "timestamp": # Skip first line
                continue
            terms = line.split(',')
            if not len(terms) == 8:
                logger.error(f"The line '{line}' doesn't contain 8 terms as expected in this format.")
                raise ValueError(f"The line '{line}' doesn't contain 8 terms as expected in this format.")
            name = terms[0]
            t = np.array(terms[1:4], dtype=float)
            q_wxyz = np.array([terms[7], terms[4], terms[5], terms[6]], dtype=float) # q = (w, x, y, z)

            # Get pose
            pose = Pose.from_quat_wxyz(q_wxyz, t)

            # Append information to the pose (augmented pose) GOOD IDEA ??? TODO
            pose.name = name

            # Append the camera pose
            poses.append(pose)

            # Print the camera pose
            logger.debug(f"'{name}': with pose:\n{pose}\n")

        logger.info(f"Successfully read {len(poses)} camera poses from file '{file}'.")
        return poses


def read_poses_from_colmap_text_file(file: Union[str, Path]) -> List[Pose]:

    # Convert to Path
    file = Path(file)

    # Input file checks
    if not file.is_file():
        logger.error(f"The input file '{str(file)}' doesn't exist.")
        raise FileExistsError(f"The input file '{str(file)}' doesn't exist.")
    if not file.suffix == ".txt":
        logger.error(f"The input file '{file}' is not a .txt file.")
        raise TypeError(f"The input file '{file}' is not a .txt file.")

    # Camera poses array
    poses: List[Pose] = []

    # Open the input text file
    with open(file) as f:

        while True:

            # Read line
            line = f.readline()
            if not line: # End of the file
                break
            if line[0] == '#': # Skip commented lines
                continue

            # Extract information
            # 'colmap' format
            terms = line.split(' ')
            if not len(terms) == 10:
                logger.error(f"The line '{line}' doesn't contain 10 terms as expected in this format.")
                raise ValueError(f"The line '{line}' doesn't contain 10 terms as expected in this format.")
            image_id = int(terms[0])
            q_wxyz = np.array(terms[1:5], dtype=float) # q = (w, x, y, z)
            t = np.array(terms[5:8], dtype=float)
            camera_id = int(terms[8])
            name = terms[9][:-1]
            # Skip next line since it contains the 2d points array
            f.readline()

            # Get pose
            # Note: the text file contains the transformation from world to camera. Thus, we need to take the
            # inverse transformation, see https://colmap.github.io/format.html#text-format
            pose = Pose.from_quat_wxyz(q_wxyz, t).inverse

            # Append information to the pose (augmented pose) GOOD IDEA ??? TODO
            pose.name = name

            # Append the camera pose
            poses.append(pose)

            # Print the camera pose
            logger.debug(f"'{name}': with pose:\n{pose}\n")

        logger.info(f"Successfully read {len(poses)} camera poses from file '{file}'.")
        return poses


def read_transformation_matrix(file: Path) -> NDArray:
    assert type(file) == pathlib.PosixPath and "The provided file is not of type Path."

    # Input file checks
    if not file.exists():
        logger.error(f"The input file '{str(file)}' doesn't exist.")
        raise FileExistsError(f"The input file '{str(file)}' doesn't exist.")
    if not file.suffix == ".txt":
        logger.error(f"The input file '{str(file)}' is not a .txt file.")
        raise TypeError(f"The input file '{str(file)}' is not a .txt file.")

    T = np.zeros((4,4))

    # Open the input text file
    with open(file) as f:

        i = 0
        while True:

            # Read line
            line = f.readline().rstrip("\n")
            if not line: # End of the file
                break

            if i >= 4:
                logger.error(f"There should not be more than 4 full lines")
                raise ValueError(f"There should not be more than 4 full lines")

            # Extract information
            terms = line.split()
            if not len(terms) == 4:
                logger.error(f"The line '{line}' doesn't contain 4 values as expected but {len(terms)} terms: " +
                                 f"{terms}")
                raise ValueError(f"The line '{line}' doesn't contain 4 values as expected but {len(terms)} terms: " +
                                 f"{terms}")

            T[i, :] = np.array(terms, dtype=float)
            i = i + 1

    return T


if COLMAP_AVAILABLE:
    def read_poses_from_colmap_model(path: Union[str, Path]) -> List[Pose]:
       """
       Returns all the camera poses contained in the provided colmap model.

       Inputs
       - colmap_model_path: path of the colmap folder containing the model. This folder should contain the .bin files
                            cameras, images, points3D.")

       Returns
       - poses:             array containing all the extracted camera poses


       Note: the poses describe the transformation from camera to world, that is t = (tx, ty, tz) are the camera
             coordinates in the world frame.
       """

       # Convert to Path
       path = Path(path)

       # Load the COLMAP model
       try:
           reconstruction = pycolmap.Reconstruction(str(path))
       except Exception as e:
           logger.error(f"Could not open the colmap model at '{str(path)}'. The error: {e}.")
           raise FileExistsError(f"Could not open the colmap model at '{str(path)}'. The error: {e}.")

       logger.info(f"Successfully imported colmap model from '{str(path)}'.")
       logger.debug(reconstruction.summary())

       # Loop through every image in the model
       camera_poses = []
       for image_id, image in reconstruction.images.items():
           logger.debug(f"Read image {image_id}: {image}")

           # Read camera pose
           camera_pose = Pose.from_colmap_image(image)
           camera_pose.name = image.name
           camera_poses.append(camera_pose)

       logger.info(f"Successfully read {len(camera_poses)} image poses the colmap model '{path}'.")

       return camera_poses


if ROS_AVAILABLE:
    def save_poses_to_new_bag(bag_name: Union[str, Path] = "./out/poses.bag", *args) -> None:
        """
        Saves the given poses to a new ROS bag file under the given topic.

        Example usage: save_poses_to_new_bag(".out/poses.bag", poses_1, topic_1, poses_2, topic_2)
        """

        # Handle the input arguments
        poses_list, topics_list  = [], []
        for arg in args:
            if type(arg) == list:
                poses_list.append(arg)
            elif type(arg) == str:
                topics_list.append(arg)

        if not len(poses_list) == len(topics_list):
            logger.error(f"Got a different number of poses and topics, {len(poses_list)} poses lists " +
                         f"and {len(topics_list)} topic names.")
            raise ValueError(f"Got a different number of poses and topics, {len(poses_list)} poses lists " +
                             f"and {len(topics_list)} topic names.")

        # Checks if ROS bag file already exists
        if Path(bag_name).exists():
            logger.info(f"ROS bag '{bag_name}' already exists and will be overwritten.")

        # Create new bag file
        with rosbag.Bag(bag_name, 'w') as bag:

            for poses, topic in zip(poses_list, topics_list):

                # Define a ROS pose array
                pose_array = PoseArray()
                pose_array.header = Header(seq=0, stamp=None, frame_id="map")

                # Loop
                pose: Pose
                for pose in poses:
                    # Append the ROS pose to the pose array
                    pose_array.poses.append(pose.to_ros_pose())

                # Save the pose array to the bag
                bag.write(topic=topic, msg=pose_array, t=None)

            logger.info(f"Finished writing to ROS bag '{bag_name}', which will contain:")
            for poses, topic in zip(poses_list, topics_list):
                logger.info(f"  - {len(poses)} poses in topic '{topic}'.")

        return