#!/usr/bin/env python3

# Python
import logging
import os.path
from pathlib import Path
from tqdm import tqdm
import numpy as np

# ROS
import rosbag
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseArray

# Src
from sba.tree_parser import TreeParser
from sba.read_poses_utils import read_poses
from sba.cylinder import Cylinder

# Utils
from sba.utils.pose import Pose
from sba.utils.loggers import get_logger
from sba.utils.folder import create_folder

def main(args):

    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.WARNING)
    logger = get_logger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    folder = Path(args.folder)
    if not folder.exists():
        logger.error(f"The provided folder '{folder}' does not exist.")
        return

    optim_steps = folder / "run" / "optim_steps"
    if not optim_steps.exists():
        logger.error(f"The provided folder '{folder}' does have a '/run/optim_steps' subfolder.")
        return

    # Find folders
    step_folders = []
    i = 0
    while (optim_steps / f"step_{i}").exists():
        step_folders.append(optim_steps / f"step_{i}")
        # Next step
        i = i + 1

    logger.info(f"Found {len(step_folders)} optimization step(s).")

    with rosbag.Bag(args.output_bag, "w") as bag:

        # Loop through each step
        for i, step_folder in tqdm(enumerate(step_folders), total=len(step_folders), desc="Steps"):

            # Timestamp
            timestamp = rospy.Time.from_sec(i/5 + 1)

            # Read camera poses
            camera_poses = read_poses(step_folder, "colmap_model")

            # Write camera poses
            pose_array = PoseArray()
            pose_array.header = Header(seq=0, stamp=None, frame_id="map")
            camera_pose: Pose
            for camera_pose in camera_poses:
                pose_array.poses.append(camera_pose.to_ros_pose())
            # Save the ROS pose array to the bag
            bag.write(topic="/camera_poses", msg=pose_array, t=timestamp)

            # Read cylinders (if exist)
            try:
                cylinders = Cylinder.from_text_file(step_folder / "cylinders.txt")
            except:
                cylinders = []

            # Write cylinders
            OPACITY = 0.7
            for cylinder in cylinders:
                cylinder_msg = cylinder.to_ros_marker(color=np.array([234 / 256, 234 / 256, 0, OPACITY]))
                bag.write(topic="/cylinders", msg=cylinder_msg, t=timestamp)

    logger.info(f"Finished writing to ROS bag '{args.output_bag}'.")
    logger.info(f"The ROS bag '{args.output_bag}' will contain:")
    logger.info(f"  - {len(step_folders)} messages with {len(camera_poses)} camera poses in the topic '/poses'.")
    logger.info(f"  - {len(step_folders)} messages with {len(cylinders)} cylinders in the topic '/cylinders'.")


if __name__ == "__main__":

    parser = TreeParser(
        description="Visualize the output of the geometric semantic bundle adjustment."
    )

    parser.add_argument("--folder", type=str, required=True, default=None,
                         help='The visualization output folder')

    parser.add_argument("-o", "--output_bag", type=str, required=False, default="out/last_optim_steps.bag",
                        help="Output bag containing the camera poses and cylinders.")

    args = parser.parse_args()

    main(args)