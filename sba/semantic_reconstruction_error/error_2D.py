# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import List, Tuple
from tqdm import tqdm

# Src
from sba.h5_utils import h5_extract
from sba.match_poses_with_h5 import check_camera_poses_and_h5_files_order
from sba.constants import semantic_reassign, SemanticError
from sba.semantic_reconstruction_error.utils import compare_semantics

# Utils
from sba.utils.pose import Pose
from sba.utils.cameras import PinholeCamera, get_homogeneous_pixels_array

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

# TODO
def ooold_but_visualization_compute_semantic_reconstruction_error(K: NDArray, camera_poses: List[Pose], h5_files: List[str],
                                                                  depth_error_threshold: float = 0.1) -> float:
    """
    Computes and prints the pixel-wise semantic reconstruction error when the pixels are backprojected onto all images.
    That is, for image i, we take all the pixels, backproject them to 3D using the given depth. Then, we project these
    3D points to all the other images j and check whether the semantic class in image j at the reprojected location is
    the same as the original semantic class in image i.

    Inputs
    - K:              (3, 3) camera intrinsics matrix of ALL the cameras
    - camera_poses:   list of the camera poses
    - h5_files:       list of the .h5 files containing the color image, the depth map and the semantic segmentation
    - depth_error_threshold: the threshold in meters that determines whether a reprojected pixel should be kept or not.
                             That is, once the 3D points are reprojected, we compare the reprojected depth (i.e. the
                             distance from image j) and the original depth form image j. If this difference is higher
                             than the threshold, this reprojected pixel/point is filtered out.

    Outputs
    - semantic_error: the overall semantic error percentage, i.e. #error_pixels / #total_pixels
    """

    if not check_camera_poses_and_h5_files_order(camera_poses, h5_files):
        logger.error("The given camera poses and h5 files don't seem to be well ordered.")
        raise ValueError("The given camera poses and h5 files don't seem to be well ordered.")

    errors: List[Tuple[Tuple[str], Tuple[float]]] = []

    # Loop through the images from which we backproject
    for i in tqdm(range(len(camera_poses)), desc="Semantic error"):
        base_camera_pose, base_h5_file = camera_poses[i], h5_files[i]

        _, base_depth_img, base_semantic_img =  h5_extract(base_h5_file)

        base_pinhole_camera = PinholeCamera(K, base_camera_pose)

        pixels_i = get_homogeneous_pixels_array(base_depth_img.shape) # (H, W, 3)

        points_world = base_pinhole_camera.backproject_to_world_using_depth(pixels_i, base_depth_img) # (H, W, 3)

        # Filter out points whose depth is negative
        depth_mask = base_depth_img > 0 # (H, W)
        pixels_i = pixels_i[depth_mask] # (?, 3)
        points_world = points_world[depth_mask] # (?, 3)
        logger.debug(f"Filtered out {np.sum(~depth_mask)} pixels (out of {depth_mask.size}, " +
                     f"i.e. {np.sum(~depth_mask)/depth_mask.size*100:.1f}%) due to negative original depth values.")

        # Loop through the images to reproject onto
        for j in tqdm(range(i + 1, len(camera_poses)), desc = "Projecting", leave=False):

            pixels_i_j = pixels_i.copy()

            camera_pose, h5_file = camera_poses[j], h5_files[j]
            _, depth_img, semantic_img = h5_extract(h5_file)

            # Don't reproject on the same camera
            if camera_pose.name == base_camera_pose.name:
                continue

            pinhole_camera = PinholeCamera(K, camera_pose)

            reprojected_pixels, reprojected_depth = pinhole_camera.project(points_world, return_depth=True) # (?, 2), (?,)

            # Round reprojected pixels values
            reprojected_pixels = np.round(reprojected_pixels).astype(int)

            # Filter out the reprojected pixels which are out of the image bounds
            bounds_mask = (reprojected_pixels[:, 0] >= 0) & (reprojected_pixels[:, 0] < semantic_img.shape[1]) & \
                          (reprojected_pixels[:, 1] >= 0) & (reprojected_pixels[:, 1] < semantic_img.shape[0])
            pixels_i_j = pixels_i_j[bounds_mask] # (?', 3)
            reprojected_pixels = reprojected_pixels[bounds_mask] # (?', 2)
            reprojected_depth = reprojected_depth[bounds_mask] # (?')
            logger.debug(f"Filtered out {np.sum(~bounds_mask)} pixels (out of {bounds_mask.size}, " +
                         f"i.e. {np.sum(~bounds_mask)/bounds_mask.size*100:.1f}%) due to reprojected pixels lying outside of the image bounds.")

            # Filter out pixels which have negative reprojected depth values
            reprojected_depth_mask = reprojected_depth > 0 # (?'')
            pixels_i_j = pixels_i_j[reprojected_depth_mask] # (?'', 3)
            reprojected_pixels = reprojected_pixels[reprojected_depth_mask] # (?'') 2)
            reprojected_depth = reprojected_depth[reprojected_depth_mask] # (?'')
            logger.debug(f"Filtered out {np.sum(~reprojected_depth_mask)} pixels (out of {reprojected_depth_mask.size}, " +
                         f"i.e. {np.sum(~reprojected_depth_mask)/reprojected_depth_mask.size*100:.1f}%) due to negative reprojected depth values.")


            # VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZ
            """
            max_depth_value = max(np.max(depth_img), np.max(reprojected_depth), np.max(base_depth_img))
            ma_depth_img = ma.masked_array(depth_img, depth_img <= 0)

            ma_reprojected_depth = ma.masked_all_like(depth_img)
            ma_reprojected_depth.data[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = reprojected_depth
            ma_reprojected_depth.mask[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = False

            def ma_to_rgba(masked_array: MaskedArray, fill_color: NDArray = np.array([255, 0, 0, 255])) -> NDArray:
                assert np.max(masked_array) <= 1 and np.min(masked_array) >= 0
                colormap = matplotlib.cm.get_cmap('viridis')
                full_array = masked_array.filled(np.min(masked_array)) # any value works
                img = (colormap(full_array) * 255).astype(np.uint8)
                img[masked_array.mask] = fill_color
                return img

            def plt_ma_show(masked_array: MaskedArray, fill_color: NDArray = np.array([255, 0, 0, 255])) -> None:
                plt.imshow(ma_to_rgba(masked_array, fill_color))

            plt.ion()
            plt.figure()
            plt_ma_show(ma.masked_array(base_depth_img/max_depth_value, base_depth_img <= 0))
            plt.title("Base depth image")

            plt.figure()
            ma_base_depth_img_used = ma.masked_array(base_depth_img, True)
            ma_base_depth_img_used.mask[pixels_i_j[:, 1], pixels_i_j[:, 0]] = False
            plt_ma_show(ma_base_depth_img_used/max_depth_value)
            plt.title("Base depth image with used pixels")

            plt.figure()
            plt_ma_show(ma.masked_array(depth_img/max_depth_value, depth_img <= 0))
            plt.title("Depth image")

            plt.figure()
            plt_ma_show(ma_reprojected_depth/max_depth_value)
            plt.title("Reprojected depth")

            plt.figure()
            ma_depth_error = np.abs(ma_reprojected_depth - ma_depth_img)
            plt_ma_show(ma_depth_error/np.max(ma_depth_error))
            plt.title("Depth error")
            """
            # VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZ

            # Get the corresponding depth of this image at the location of the reprojected pixels
            depth = depth_img[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] # (?'',)

            # Filter out pixels whose reprojected depth is not close enough to the given depth for this image
            depth_error = np.abs(depth - reprojected_depth) # (?'',)
            depth_error_mask = depth_error <= depth_error_threshold
            # Apply mask
            pixels_i_j = pixels_i_j[depth_error_mask] # (?''', 3)
            reprojected_pixels = reprojected_pixels[depth_error_mask] # (?''') 2)
            reprojected_depth = reprojected_depth[depth_error_mask] # (?''',)
            logger.debug(f"Filtered out {np.sum(~depth_error_mask)} pixels (out of {depth_error_mask.size}, " +
                         f"i.e. {np.sum(~depth_error_mask)/depth_error_mask.size*100:.1f}%) due to high depth errors (> {depth_error_threshold}).")

            # Compare semantics
            base_semantics = base_semantic_img[pixels_i_j[:, 1], pixels_i_j[:, 0]] # (?''',)
            semantics = semantic_img[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] # (?''',)
            semantic_error = semantic_reassign(semantics) != semantic_reassign(base_semantics) # (?''',)

            # TODO confusion matrix !!!!!!

            # - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION -
            """
            ma_reprojected_depth = ma.masked_all_like(depth_img)
            ma_reprojected_depth.data[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = reprojected_depth
            ma_reprojected_depth.mask[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = False
            plt.figure()
            plt_ma_show(ma_reprojected_depth/max_depth_value)
            plt.title("Reprojected depth (after filtering out points with large depth error)")

            ma_semantic_error = ma.masked_all_like(depth_img)
            ma_semantic_error.data[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = semantic_error
            ma_semantic_error.mask[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] = False
            plt.figure()
            plt_ma_show(ma_semantic_error)
            plt.title("Semantic error")
            """
            # - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION - VISUALIZATION -

            logger.debug(f"**When reprojecting from {base_camera_pose.name} onto {camera_pose.name}: " +
                         f"{np.sum(semantic_error)/semantic_error.size*100:.1f}% of the pixels " +
                         f"(out of {semantic_error.size:.1e}) have wrong semantic classes. **")

            errors.append(((base_camera_pose.name, camera_pose.name), (np.sum(semantic_error), semantic_error.size)))

    # Output
    max_name_length = max([len(c.name) for c in camera_poses])
    logger.info(f"**The semantic reconstruction error is as follows:**")
    for (base_name, reprojected_name), (n_error, n_pixels) in errors:
        logger.info(f"- {base_name:{max_name_length}} -> {reprojected_name:{max_name_length}}: " +
                    f"{n_error:.2e} pixels have wrong semantic classes out of {n_pixels:.2e}, " +
                    f"i.e. {n_error/n_pixels*100:2.1f}%.")

    # Output globally
    total_n_error = sum([e[1][0] for e in errors])
    total_n_pixels = sum([e[1][1] for e in errors])
    logger.info(f"** Globally, that means about {total_n_error/total_n_pixels*100:.1f}% pixels have wrong semantic classes when they are reprojected.**")

    return total_n_error/total_n_pixels


def semantic_2D_reconstruction_error_between_two_images(base_camera: PinholeCamera, base_h5_file: str,
                                                        camera: PinholeCamera, h5_file: str,
                                                        depth_error_threshold: float = 0.1) -> SemanticError:
    """
    Computes the 2D pixel-wise semantic reconstruction error when the pixels of the base image are backprojected to the
    other image. Here are the steps:
    - all the pixels of the base image are backprojected to 3D using the given depth
    - then, these 3D points are reprojected to the second provided image
    - finally, the semantic classes of the base image and the ones of the second image are compared
    - these comparisons occur at the reprojected locations on the second image

    Inputs
    - K:              (3, 3) camera intrinsics matrix of ALL the cameras
    - base_camera:    Pose with a name attribute, camera pose of the base image
    - base_h5:        h5 file containg the color image, the depth map and the semantic segmentation of the base image
    - camera:         Pose with a name attribute, camera pose of the second image
    - h5_file:        h5 file containg the color image, the depth map and the semantic segmentation of the second image
    - depth_error_threshold: the threshold in meters that determines whether a reprojected pixel should be kept or not.
                             That is, once the 3D points are reprojected, we compare the reprojected depth (i.e. the
                             distance from image j) and the original depth form image j. If this difference is higher
                             than the threshold, this reprojected pixel/point is filtered out.

    Outputs
    - semantic_error: SemanticError = ((base_camera.name, camera.name), (np.sum(error), error.size))
    """

    # Check that the pinhole cameras have a 'name' attribute
    if not hasattr(base_camera, "name") or not hasattr(camera, "name"):
        logger.error("The PinholeCameras don't seem to have a 'name' attribute.")
        raise ValueError("The PinholeCameras don't seem to have a 'name' attribute.")

    # Acertain not reprojecting on the same camera
    if camera.name == base_camera.name:
        logger.error(f"The givne images pair implies reprojecting image '{base_camera.name}' onto '{camera.name}'. " +
                     f"Returning a 'None' error.")
        raise ValueError(
            f"The given images pair implies reprojecting image '{base_camera.name}' onto '{camera.name}'. " +
            f"Returning a 'None' error.")

    # None error
    none_error: SemanticError = ((base_camera.name, camera.name), (None, None))

    def zero_quit(value: int) -> bool:
        if value == 0:
            logger.warning(f"When projecting {base_camera.name} onto {camera.name}: there are no pixels left for " +
                           f"computing the semantic reconstruction error. Returning a 'None' error.")
            return True
        return False

    # Extract the h5_files
    _, base_depth_img, base_semantic_img = h5_extract(base_h5_file)
    _, depth_img, semantic_img = h5_extract(h5_file)

    base_pixels = get_homogeneous_pixels_array(base_depth_img.shape) # (H, W, 3)

    points_world = base_camera.backproject_to_world_using_depth(base_pixels, base_depth_img) # (H, W, 3)

    # Filter out points whose depth is negative
    depth_mask = base_depth_img > 0 # (H, W)
    base_pixels = base_pixels[depth_mask] # (?, 3)
    points_world = points_world[depth_mask] # (?, 3)
    if zero_quit(len(points_world)): return none_error
    logger.debug(f"Filtered out {np.sum(~depth_mask)} pixels (out of {depth_mask.size}, " +
                 f"i.e. {np.sum(~depth_mask)/depth_mask.size*100:.1f}%) due to negative original depth values.")

    # Reproject the pixels onto the second camera frame
    reprojected_pixels, reprojected_depth = camera.project(points_world, return_depth=True) # (?, 2), (?,)

    # Round reprojected pixels values
    reprojected_pixels = np.round(reprojected_pixels).astype(int)

    # Filter out the reprojected pixels which are out of the image bounds
    bounds_mask = (reprojected_pixels[:, 0] >= 0) & (reprojected_pixels[:, 0] < semantic_img.shape[1]) & \
                  (reprojected_pixels[:, 1] >= 0) & (reprojected_pixels[:, 1] < semantic_img.shape[0])
    base_pixels = base_pixels[bounds_mask] # (?', 3)
    reprojected_pixels = reprojected_pixels[bounds_mask] # (?', 2)
    reprojected_depth = reprojected_depth[bounds_mask] # (?')
    if zero_quit(len(reprojected_depth)): return none_error
    logger.debug(f"Filtered out {np.sum(~bounds_mask)} pixels (out of {bounds_mask.size}, " +
                 f"i.e. {np.sum(~bounds_mask)/bounds_mask.size*100:.1f}%) due to " +
                 f"reprojected pixels lying outside of the image bounds.")

    # Filter out pixels which have negative reprojected depth values
    reprojected_depth_mask = reprojected_depth > 0 # (?'')
    base_pixels = base_pixels[reprojected_depth_mask] # (?'', 3)
    reprojected_pixels = reprojected_pixels[reprojected_depth_mask] # (?'') 2)
    reprojected_depth = reprojected_depth[reprojected_depth_mask] # (?'')
    if zero_quit(len(reprojected_depth)): return none_error
    logger.debug(f"Filtered out {np.sum(~reprojected_depth_mask)} pixels (out of {reprojected_depth_mask.size}, " +
                    f"i.e. {np.sum(~reprojected_depth_mask)/reprojected_depth_mask.size*100:.1f}%) " +
                    f"due to negative reprojected depth values.")

    # Get the corresponding depth of this image at the location of the reprojected pixels
    depth = depth_img[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] # (?'',)

    # Filter out pixels whose reprojected depth is not close enough to the given depth for this image
    depth_error = np.abs(depth - reprojected_depth) # (?'',)
    depth_error_mask = depth_error <= depth_error_threshold
    # Apply mask
    base_pixels = base_pixels[depth_error_mask] # (?''', 3)
    reprojected_pixels = reprojected_pixels[depth_error_mask] # (?''') 2)
    reprojected_depth = reprojected_depth[depth_error_mask] # (?''',)
    if zero_quit(len(reprojected_depth)): return none_error
    logger.debug(f"Filtered out {np.sum(~depth_error_mask)} pixels (out of {depth_error_mask.size}, " +
                    f"i.e. {np.sum(~depth_error_mask)/depth_error_mask.size*100:.1f}%) due to high depth errors (> {depth_error_threshold}).")

    # Compare semantics
    base_semantics = base_semantic_img[base_pixels[:, 1], base_pixels[:, 0]] # (?''',)
    semantics = semantic_img[reprojected_pixels[:, 1], reprojected_pixels[:, 0]] # (?''',)

    error: SemanticError = compare_semantics(base_semantics, semantics, base_camera, camera)
    return error