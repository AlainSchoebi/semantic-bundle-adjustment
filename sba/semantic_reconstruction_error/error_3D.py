# Numpy
import numpy as np

# Scipy
from scipy.spatial import cKDTree

# Src
from sba.h5_utils import h5_extract
from sba.constants import SemanticError
from sba.semantic_reconstruction_error.utils import compare_semantics

# Utils
from sba.utils.pose import Pose
from sba.utils.cameras import PinholeCamera, get_homogeneous_pixels_array

# Logging
from sba.utils.loggers import get_logger

logger = get_logger(__name__)


def semantic_3D_reconstruction_error_between_two_images(
        base_camera: PinholeCamera,
        base_h5_file: str,
        camera: PinholeCamera,
        h5_file: str,
        distance_error_threshold: float = 0.1) -> SemanticError:
    """
    Computes the 3D pixel-wise semantic reconstruction error when the pixels of the base image are backprojected to 3D.
    Here are the main steps:
    - all the pixels of the base image are backprojected to 3D using the given depth
    - all the pixels of the second image are backprojected to 3D using the given depth
    - then, for every 3D point of the base image, the closest 3D point of the second image is found
    - finally, the semantic classes of the base image and the ones of the second image are compared
    - these comparisons happen between the 3D point of the base image and the closest 3D point of the second image

    Inputs
    - K:              (3, 3) camera intrinsics matrix of ALL the cameras
    - base_camera:    Pose with a name attribute, camera pose of the base image
    - base_h5:        h5 file containg the color image, the depth map and the semantic segmentation of the base image
    - camera:         Pose with a name attribute, camera pose of the second image
    - h5_file:        h5 file containg the color image, the depth map and the semantic segmentation of the second image
    - distance_error_threshold: the threshold in meters that determines whether a match between two 3D points should be
                                kept or not. That is, if the distance between two matched 3D points is larger
                                than the threshold, the error for the corresponding base pixel is filtered out.

    Outputs
    - semantic_error: SemanticError = ((base_camera.name, camera.name), (np.sum(error), error.size))
    """

    # Check that the pinhole cameras have a 'name' attribute
    if not hasattr(base_camera, "name") or not hasattr(camera, "name"):
        logger.error(
            "The PinholeCameras don't seem to have a 'name' attribute.")
        raise ValueError(
            "The PinholeCameras don't seem to have a 'name' attribute.")

    # Acertain not reprojecting on the same camera
    if camera.name == base_camera.name:
        logger.error(
            f"The givne images pair implies reprojecting image '{base_camera.name}' onto '{camera.name}'. "
            + f"Returning a 'None' error.")
        raise ValueError(
            f"The givne images pair implies reprojecting image '{base_camera.name}' onto '{camera.name}'. "
            + f"Returning a 'None' error.")

    # None error
    none_error: SemanticError = ((base_camera.name, camera.name), (None, None))

    def zero_quit(value: int) -> bool:
        if value == 0:
            logger.warning(
                f"When projecting {base_camera.name} onto {camera.name}: there are no pixels left for "
                +
                f"computing the semantic reconstruction error. Returning a 'None' error."
            )
            return True
        return False

    # Downsampling rate
    DS = 10

    # Extract the h5_files
    _, depth_img_1, semantic_img_1 = h5_extract(base_h5_file)
    _, depth_img_2, semantic_img_2 = h5_extract(h5_file)

    # Project the base image pixels to 3D
    pixels_1 = get_homogeneous_pixels_array(depth_img_1.shape)  # (H, W, 3)

    # Downsampling
    pixels_1 = pixels_1[::DS, ::DS, :]
    depth_img_1 = depth_img_1[::DS, ::DS]

    # Back projecting to 3D
    points_world_1 = base_camera.backproject_to_world_using_depth(
        pixels_1, depth_img_1)  # (H, W, 3)

    # Filter out points whose depth is negative
    depth_mask_1 = depth_img_1 > 0  # (H, W)
    pixels_1 = pixels_1[depth_mask_1]  # (?, 3)
    points_world_1 = points_world_1[depth_mask_1]  # (?, 3)
    if zero_quit(len(points_world_1)): return none_error
    logger.debug(
        f"Filtered out {np.sum(~depth_mask_1)} pixels (out of {depth_mask_1.size}, "
        +
        f"i.e. {np.sum(~depth_mask_1)/depth_mask_1.size*100:.1f}%) due to negative original depth values."
    )

    # Project the second image pixels to 3D
    pixels_2 = get_homogeneous_pixels_array(depth_img_2.shape)  # (H, W, 3)

    # Downsampling
    pixels_2 = pixels_2[::DS, ::DS, :]
    depth_img_2 = depth_img_2[::DS, ::DS]

    # Back projecting to 3D
    points_world_2 = camera.backproject_to_world_using_depth(
        pixels_2, depth_img_2)  # (H, W, 3)

    # Filter out points whose depth is negative
    depth_mask_2 = depth_img_2 > 0  # (H, W)
    pixels_2 = pixels_2[depth_mask_2]  # (?, 3)
    points_world_2 = points_world_2[depth_mask_2]  # (?, 3)
    if zero_quit(len(points_world_2)): return none_error
    logger.debug(
        f"Filtered out {np.sum(~depth_mask_2)} pixels (out of {depth_mask_2.size}, "
        +
        f"i.e. {np.sum(~depth_mask_2)/depth_mask_2.size*100:.1f}%) due to negative original depth values."
    )

    # Build KD-Tree for the second point cloud
    kdtree_2 = cKDTree(points_world_2)

    # Find the closest point in the second cloud for each point in the first cloud
    query = kdtree_2.query(points_world_1, k=1)
    distances = query[0]
    indices = query[1]

    # Get the corresponding pixels
    corresponding_pixels_2 = pixels_2[indices]

    # Apply distance error threshold
    mask = distances < distance_error_threshold
    pixels_1_valid = pixels_1[mask]
    corresponding_pixels_2_valid = corresponding_pixels_2[mask]

    # (optional) corresponding 3D points
    corresponding_points_world_2 = points_world_2[indices]
    points_world_1_valid = points_world_1[mask]
    points_world_2_valid = corresponding_points_world_2[mask]

    if zero_quit(len(corresponding_pixels_2_valid)): return none_error

    # Get the coresponding semantics
    semantics_1 = semantic_img_1[pixels_1_valid[:, 1], pixels_1_valid[:, 0]]
    semantics_2 = semantic_img_2[corresponding_pixels_2_valid[:, 1],
                                 corresponding_pixels_2_valid[:, 0]]

    # Compare the semantics
    error: SemanticError = compare_semantics(semantics_1, semantics_2,
                                             base_camera, camera)
    return error
