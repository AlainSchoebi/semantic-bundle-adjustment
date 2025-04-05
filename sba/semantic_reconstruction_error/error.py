# Numpy
import numpy as np
from numpy.typing import NDArray
import numpy.ma as ma
from numpy.ma import MaskedArray

# Python
from typing import List, Tuple, Type, Literal
from tqdm import tqdm

# Src
from sba.match_poses_with_h5 import check_camera_poses_and_h5_files_order
from sba.constants import semantic_reassign, SemanticError
from sba.semantic_reconstruction_error.error_2D import semantic_2D_reconstruction_error_between_two_images
from sba.semantic_reconstruction_error.error_3D import semantic_3D_reconstruction_error_between_two_images

# Utils
from sba.utils.pose import Pose
from sba.utils.cameras import PinholeCamera

# Logging
from sba.utils.loggers import get_logger

logger = get_logger(__name__)


def compare_semantics(
    semantics_1: NDArray,
    semantics_2: NDArray,
    camera_1: PinholeCamera,
    camera_2: PinholeCamera,
) -> SemanticError:
    """
    Compare two semantics vector, where semantics_1 is considered as the base
    vector and semantics_2 the vector to compare to.
    """

    # Reassign the semantic classes if needed
    semantics_1, semantics_2 = semantic_reassign(
        semantics_1), semantic_reassign(semantics_2)

    # Confusion matrix
    classes = np.unique(
        np.concatenate((np.unique(semantics_1),
                        np.unique(semantics_2))))  # C unique classes
    # (N, C) N: number of pixels, C: number of classes
    mask = semantics_1[:, None] == classes
    mm = ma.array(semantics_2[:, None][:, [0] * len(classes)],
                  mask=~mask)  # (N, C)
    corresponding_classes = mm[..., None] == classes  # (N, C, C)
    confusion_matrix = ma.sum(corresponding_classes, axis=0).filled(0)

    # Compute error
    semantic_error = semantics_2 != semantics_1  # (?''',)

    # Log
    logger.debug(
        f"**When reprojecting from {camera_1.name} onto {camera_2.name}: " +
        f"{np.sum(semantic_error)/semantic_error.size*100:.1f}% of the pixels " +
        f"(out of {semantic_error.size:.1e}) have wrong semantic classes. **")
    logger.debug(f"The confusion matrix, for the classes {classes}, is given by:\n{confusion_matrix}\n")

    error: SemanticError = (
        (camera_1.name, camera_2.name),
        (np.sum(semantic_error), semantic_error.size),
    )
    return error


def compute_semantic_reconstruction_error_from_indices_pairs(
    K: NDArray,
    camera_poses: List[Pose],
    h5_files: List[str],
    indices: List[Tuple[int]],
    error_type: Literal["2D", "3D"] = "2D",
    depth_error_threshold: float = 0.1,
    distance_error_threshold: float = 0.1,
) -> Tuple[int, int]:
    """
    Compute the semantic reconstructino error between every pair of image given by the indices.
    Inputs
    - K:              (3, 3) camera intrinsics matrix of ALL the cameras
    - camera_poses:   list of the camera poses
    - h5_files:       list of the .h5 files containing the color image, the depth map and the semantic segmentation.
    - indices:        list of pair of integers, representing the pair of images for which the error is computed.
    - error_type:     determines the type of semantic error computation, i.e. computed in 2D or in 3D
    - depth_error_threshold: only relevant for 2D error computation. The threshold in meters that determines whether a
                             reprojected pixel should be kept or not. That is, once the 3D points are reprojected, we
                             compare the reprojected depth (i.e. the distance from image j) and the original depth from
                             image j. If this difference is higher than the threshold, this reprojected pixel/point is
                             filtered out.
    - distance_error_threshold: only relevant for 3D error computation. The threshold in meters that determines whether
                                a match between two 3D points should be kept or not. That is, if the distance between
                                two matched 3D points is larger than the threshold, the error for the corresponding base
                                pixel is filtered out.

    Outputs
    - total_n_error:  the total number of pixels for which the semantic classes do not correspond
    - total_n_pixel:  the total number of pixels for which the semantic error was computed
    """

    if not check_camera_poses_and_h5_files_order(camera_poses, h5_files):
        logger.error(
            "The given camera poses and h5 files don't seem to be well ordered."
        )
        raise ValueError(
            "The given camera poses and h5 files don't seem to be well ordered."
        )

    errors: List[Tuple[Tuple[str], Tuple[int]]] = []

    # Loop through the indices pairs from which we backproject
    for i, j in tqdm(indices, desc="Semantic error"):
        # i: base image index
        # j: reprojected image index

        if not (i >= 0 and j >= 0 and i < len(camera_poses) and j < len(camera_poses)):
            logger.warning(f"The index pair ({i}, {j}) is not in bounds [0, {len(camera_poses)}]. Skipping it.")
            continue

        # Base
        base_camera_pose, base_h5_file = camera_poses[i], h5_files[i]

        base_camera = PinholeCamera(K, base_camera_pose)
        base_camera.name = base_camera_pose.name

        # Reprojection camera
        camera_pose, h5_file = camera_poses[j], h5_files[j]

        camera = PinholeCamera(K, camera_pose)
        camera.name = camera_pose.name

        # Don't reproject on the same camera
        if camera_pose.name == base_camera_pose.name:
            logger.warning(
                f"The index pair ({i}, {j}) implies reprojecting image '{base_camera_pose.name}'"
                + f"onto '{camera_pose.name}'. Skipping it.")
            continue

        if error_type != "2D" and error_type != "3D":
            logger.error(
                f"The provided error_type '{error_type}' can only be '2D' or '3D'"
            )
        if error_type == "2D":
            error = semantic_2D_reconstruction_error_between_two_images(
                base_camera,
                base_h5_file,
                camera,
                h5_file,
                depth_error_threshold=depth_error_threshold,
            )
        else:
            error = semantic_3D_reconstruction_error_between_two_images(
                base_camera,
                base_h5_file,
                camera,
                h5_file,
                distance_error_threshold=distance_error_threshold,
            )
        errors.append(error)

    # Output
    max_name_length = max([len(c.name) for c in camera_poses])
    logger.info(f"**The semantic reconstruction error is as follows:**")
    for (base_name, reprojected_name), (n_error, n_pixels) in errors:
        if n_error is not None:
            logger.info(
                f"- {base_name:{max_name_length}} -> {reprojected_name:{max_name_length}}: "
                +
                f"{n_error:.2e} pixels have wrong semantic classes out of {n_pixels:.2e}, "
                + f"i.e. {n_error/n_pixels*100:2.1f}%.")
        else:
            logger.info(
                f"- {base_name:{max_name_length}} -> {reprojected_name:{max_name_length}}: "
                + f" no error could be computed.")

    # Output globally
    total_n_error = int(sum([e[1][0] for e in errors if e[1][0] is not None]))
    total_n_pixels = int(sum([e[1][1] for e in errors if e[1][0] is not None]))
    if not total_n_pixels == 0:
        logger.info(
            f"** Globally, that means about {total_n_error/total_n_pixels*100:.1f}% pixels have wrong semantic classes when they are reprojected.**"
        )
        return total_n_error, total_n_pixels
    else:
        logger.warning(f"** Globally, no error could be computed.**")
        return None