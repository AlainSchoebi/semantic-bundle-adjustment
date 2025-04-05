#!/usr/bin/env python3

# Typing
from typing import List, Union, Optional
from numpy.typing import NDArray

# Python
import numpy as np
import logging
from tqdm import tqdm
import os.path
from pathlib import Path

# ROS
import rosbag
import rospy
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Header

# Src
from sba.constants import semantic_colors_rgba, semantic_img_to_rgb, depth_img_to_rgb, \
    camera_intrinsic_matrix, semantic_reassign
from sba.tree_parser import TreeParser
from sba.h5_utils import get_all_h5_files_in_folder, h5_extract
from sba.read_poses_utils import read_poses
from sba.match_poses_with_h5 import match_camera_poses_and_h5_files_order
from sba.cylinder import Cylinder

# Utils
from sba.utils.loggers import get_logger
from sba.utils.pose import Pose
from sba.utils.ros_pointclouds import points_to_pointcloud, points_to_marker
from sba.utils.ros_markers import image_with_camera_pose_to_marker
from sba.utils.cameras import PinholeCamera, get_homogeneous_pixels_array

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

def reconstruct_3d(
       # Required arguments
       camera_poses: List[Pose],
       h5_files: Union[List[Path], List[str]],
       K: NDArray,
       output_bag: Union[str, Path],

       # Optional arguments
       topic_poses: Optional[str] = "/camera_poses",
       topic_pointcloud: Optional[str] = "/points",
       topic_semantic_pointcloud: Optional[str] = "/semantic_points",
       topic_images: Optional[str] = "/images",
       output_npy: Optional[str] = None,
       cylinders: Optional[List[Cylinder]] = [],
       SAVE_IMAGES: Optional[bool] = True,
       FOCAL_LENGTH: float = 3,
       MAX_DEPTH: float = 50,
       MAX_LENGTH_PIXELS: int = int(200),
       MAX_POINTS_SAVE: int = int(1e6),
       MAX_SEMANTIC_POINTS_SAVE: int = int(6e5),
):

    # Check if the order has matching image names
    match_camera_poses_and_h5_files_order(camera_poses, h5_files)

    # Camera intrinsics
    assert K.shape == (3,3) and "Intrinsic matrix must have shape (3,3)."

    # Max points type
    assert type(MAX_POINTS_SAVE) == int and type(MAX_SEMANTIC_POINTS_SAVE) == int

    # Define arrays
    all_points = []
    all_semantics = []
    all_image_ids = []
    images_markers_msg = []

    # Loop
    for i, (camera_pose, h5_file) in tqdm(
        enumerate(zip(camera_poses, h5_files)),
        total=len(h5_files),
        desc="Reconstructing 3D points from .h5 files",
    ):
        color_img, depth_img, semantic_img = h5_extract(h5_file)

        # Create a PinholeCamera
        pinhole_camera = PinholeCamera(K, camera_pose)

        # Create an image marker for visualization
        if SAVE_IMAGES:
            # Parameters to chose
            # Color image
            color_img_marker_msg = image_with_camera_pose_to_marker(
                color_img,
                pinhole_camera,
                focal_length=FOCAL_LENGTH,
                max_length_pixels=MAX_LENGTH_PIXELS,
                id=3 * i,
                ns="color_images",
            )
            images_markers_msg.append(color_img_marker_msg)

            # Depth image
            depth_img_marker_msg = image_with_camera_pose_to_marker(
                depth_img_to_rgb(depth_img, MAX_DEPTH),
                pinhole_camera,
                focal_length=FOCAL_LENGTH,
                max_length_pixels=MAX_LENGTH_PIXELS,
                id=3 * i + 1,
                ns="depth_images",
            )
            images_markers_msg.append(depth_img_marker_msg)

            # Semantic image
            semantic_img_marker_msg = image_with_camera_pose_to_marker(
                semantic_img_to_rgb(semantic_img),
                pinhole_camera,
                focal_length=FOCAL_LENGTH,
                max_length_pixels=MAX_LENGTH_PIXELS,
                id=3 * i + 2,
                ns="semantic_images",
            )
            images_markers_msg.append(semantic_img_marker_msg)

        # Project the pixels to 3D using the camera poses and the depth map
        pixels = get_homogeneous_pixels_array(depth_img.shape)

        # Backproject to the world frame using the depth image
        points_world = pinhole_camera.backproject_to_world_using_depth(
            pixels, depth_img
        )  # (H, W, 3)

        # Reject any pixel whose depth is negative
        depth_img_mask = depth_img > 0  # (H, W)
        points_filtered = points_world[depth_img_mask]  # (?, 3)
        semantics_filtered = semantic_img[depth_img_mask]  # (?, )
        image_ids = np.full(len(points_filtered), fill_value=i)  # (?, )

        # Append new points to the all points array
        all_points.append(points_filtered)
        all_semantics.append(semantics_filtered)
        all_image_ids.append(image_ids)
        logger.debug(f"Successfully reconstructed {len(points_filtered)} 3D points from image '{camera_pose.name}'.")

    # Concatenate all points together
    all_points = np.concatenate(all_points, axis=0)  # (?, 3)
    all_semantics = np.concatenate(all_semantics, axis=0)  # (?, )
    all_image_ids = np.concatenate(all_image_ids, axis=0)  # (?, )

    # Reassign semantics
    all_semantics = semantic_reassign(all_semantics)

    # Save to .npy file
    if output_npy is not None:
        np.save(output_npy, all_points)
        logger.info(f"Successfully saved {len(all_points)} 3D points to file '{output_npy}'.")

    # Log
    logger.info(f"Successfully reconstructed {len(all_points)} 3D points from {len(h5_files)} depth images.")

    # Save the 3d reconstructed points to a ROS bag
    # Create new bag file
    timestamp = None
    with rosbag.Bag(output_bag, "w") as bag:
        # Camera poses
        # Define a ROS pose array
        pose_array = PoseArray()
        pose_array.header = Header(seq=0, stamp=None, frame_id="map")
        camera_pose: Pose
        for camera_pose in camera_poses:
            # Append the ROS pose to the pose array
            pose_array.poses.append(camera_pose.to_ros_pose())
        # Save the ROS pose array to the bag
        bag.write(topic=topic_poses, msg=pose_array, t=None)

        # Pointcloud
        # Define which points will be saved
        if MAX_POINTS_SAVE > len(all_points):
            points_to_save_normal = all_points
            logger.info(f"Selected all {len(all_points)} points to save.")
        else:
            indices = np.random.default_rng().choice(
                len(all_points), size=MAX_POINTS_SAVE, replace=False
            )
            points_to_save_normal = all_points[indices]
            logger.info(
                f"Randomly selected {len(points_to_save_normal)} points to save (out of {len(all_points)}) for "
                + f"the NORMAL pointcloud (saved as Pointcloud2) ({len(points_to_save_normal)/len(all_points)*100:.2f}%)"
            )

        # Build pointcloud message and save it
        points_msg = points_to_pointcloud(points_to_save_normal)
        bag.write(topic=topic_pointcloud, msg=points_msg, t=timestamp)

        # Semantic pointcloud
        # Define which points will be saved
        if MAX_SEMANTIC_POINTS_SAVE > len(all_points):
            points_to_save_semantic = all_points
            semantics_to_save = all_semantics
            image_ids_to_save = all_image_ids
            logger.info(
                f"Selected all {len(all_points)} points to save to the semantic pointcloud."
            )
        else:
            indices = np.random.default_rng().choice(
                len(all_points),
                size=MAX_SEMANTIC_POINTS_SAVE,
                replace=False,
            )
            points_to_save_semantic = all_points[indices]
            semantics_to_save = all_semantics[indices]
            image_ids_to_save = all_image_ids[indices]
            logger.info(
                f"Randomly selected {len(points_to_save_semantic)} points to save (out of {len(all_points)}) for "
                + f"the SEMANTIC pointcloud (saved as Marker) ({len(points_to_save_semantic)/len(all_points)*100:.2f}%)"
            )
        # Build marker message for the semantically annotated pointcloud
        colors = np.array([semantic_colors_rgba[semantic] for semantic in semantics_to_save])

        marker_msg = points_to_marker(points_to_save_semantic, colors, id=597387)
        bag.write(
            topic=topic_semantic_pointcloud, msg=marker_msg, t=timestamp
        )

        # Images
        for image_marker_msg in images_markers_msg:
            bag.write(topic=topic_images, msg=image_marker_msg, t=timestamp)

        # Cylinders
        for cylinder in cylinders:
            cylinder_msg = cylinder.to_ros_marker(color=np.array([234 / 256, 234 / 256, 0, 0.7]))
            bag.write(topic="cylinders", msg=cylinder_msg, t=timestamp)

        # Final info log
        logger.info(f"Finished writing to ROS bag '{output_bag}'.")
        logger.info(
            f"The ROS bag '{output_bag}' will contain:\n"
            + f"  - {len(camera_poses)} camera poses in topic '{topic_poses}'\n"
            + f"  - {len(points_to_save_normal)} points in the pointcloud in the topic '{topic_pointcloud}'\n"
            + f"  - {len(points_to_save_semantic)} semantic points in the marker topic '{topic_semantic_pointcloud}'\n"
            + f"  - {len(images_markers_msg)} images in the marker topic '{topic_images}'"
        )


def main(args):
    # Setup logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sba").setLevel(logging.INFO)
    logger.setLevel(logging.DEBUG)

    # Camera poses array
    camera_poses = read_poses(args.camera_poses_path, args.camera_poses_format)
    # Get all .h5 files contained in the h5 folder (in sorted order)
    h5_files = get_all_h5_files_in_folder(args.h5_folder)
    # Check if the order has matching image names
    match_camera_poses_and_h5_files_order(camera_poses, h5_files)

    # Take a subset of the images if asked
    if args.subset is not None:
        camera_poses = camera_poses[: args.subset]
        h5_files = h5_files[: args.subset]
        logger.info(
            f"Only using a subset of {args.subset} images and camera poses."
        )

    # Camera intrinsics
    K = camera_intrinsic_matrix(args.camera_intrinsics)

    # Cylinders
    cylinders = []
    if args.cylinder_file is not None:
        cylinders = Cylinder.from_text_file(args.cylinder_file)

    # 3D reconstruction
    reconstruct_3d(
        camera_poses=camera_poses,
        h5_files=h5_files,
        K=K,
        output_bag=args.output_bag,
        topic_poses=args.topic_poses,
        topic_pointcloud=args.topic_pointcloud,
        topic_semantic_pointcloud=args.topic_semantic_pointcloud,
        topic_images=args.topic_images,
        output_npy=args.output_npy,
        cylinders=cylinders,
        SAVE_IMAGES=args.save_images,
        FOCAL_LENGTH=3,
        MAX_DEPTH=50,
        MAX_LENGTH_PIXELS=400,
        MAX_POINTS_SAVE=int(args.max_points_save),
        MAX_SEMANTIC_POINTS_SAVE=int(args.max_semantic_points_save)
    )

if __name__ == "__main__":
    parser = TreeParser(
        description="Reconstructs the 3D environment point cloud using camera poses and depth maps."
    )

    parser.add_workspace_and_camera_h5_folder_arguments()
    parser.add_camera_intrinsics_argument()

    parser.add_argument("--subset", type=int, required=False, default=None,
        help="Take only a subset of the all the images and camera poses. Default is use all images.")

    # Additional clinders to visualize
    parser.add_argument("--cylinder_file", type=str, required=False, default=None,
                        help="Additional text file containing cylinders for visualization.")

    # Output .npy file
    parser.add_argument("--output_npy", type=str, required=False, default="out/last_3d_reconstruction.npy",
        help="Output .npy file containg the pointcloud.")

    # Point saving ratio
    parser.add_argument("--max_points_save", type=int, required=False,
                        default=1e6,
                        help="Maximum number of points being saved to the .bag file for the normal reconstruction.")
    parser.add_argument("--max_semantic_points_save", type=int, required=False,
                        default=6e5,
                        help="Maximum number of points being saved to the .bag file for the semantic reconstruction.")

    # ROS .bag file
    parser.add_argument("--output_bag", "-o", type=str, required=False,
                        default="out/last_3d_reconstruction.bag",
                        help="Output bag file containing the pointcloud.")
    parser.add_argument("--topic_poses", type=str, required=False,
                        default="/camera_poses",
                        help="Topic name in the output bag containing the camera poses.")
    parser.add_argument("--topic_pointcloud", type=str, required=False,
                        default="/points",
                        help="Topic name in the output bag containing the point cloud.")
    parser.add_argument("--topic_semantic_pointcloud", type=str, required=False,
                        default="/semantic_points",
                        help="Topic name in the output bag containing the semantic pointcloud (marker).")
    parser.add_argument("--save_images", action="store_true",
                         default=True,
                        help="Whether the 2D images should be exported to the .bag file or not.")
    parser.add_argument("--no-save_images", action="store_false", dest="save_images",
                        help="Whether the 2D images should be exported to the .bag file or not.")
    parser.add_argument("--topic_images", type=str, required=False,
                         default="/images",
                        help="Topic name in the output bag containing the images (marker).")

    args = parser.parse_args()

    main(args)