# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import Tuple, Union

# Utils
from .pose import Pose
from .base import homogenized, dehomogenized


def get_homogeneous_pixels_array(image_shape: Tuple[int]) -> NDArray:

    """
    Create a pixel array containing homogeneous coordinates (u, v, 1) for every pixel coordinate in an image.

    Inputs
    - image_shape: Tuple describing the dimensions of the image, (H, W)

    Outputs
    - pixels:      (H, W, 3) array containing the homogeneous coordinates
    """

    if not len(image_shape) == 2:
        raise ValueError(f"The provided shape must have 2 elements, i.e. (H, W), not {image_shape}.")

    pixels = np.ones((*image_shape, 3), dtype=int) # (H, W, 3) (homogeneous 2D pixel coordinates)
    pixels[:, :, 0] = np.arange(image_shape[1])[None, :]  # u values
    pixels[:, :, 1] = np.arange(image_shape[0])[:, None]  # v values

    return pixels

class PinholeCamera:

    def __init__(self, K: NDArray, pose: Pose = Pose()):
        """
        Create a PinholeCamera from the given intrinsics and extrinsics.

        Inputs
        - K:     (3,3) camera intrinsics matrix
        - pose:  Pose camera extrinsics as the transformation from camera to world. I.e. t = (tx, ty, tz) represents the
                 camera coodirnates in the world frame.
        """
        self.__pose = pose
        self.set_intrinsics(K)

    def set_intrinsics(self, K: NDArray):
        assert(K.shape == (3, 3))
        self.__K = K
        self.__K_inv = np.linalg.inv(K)
        self._compute_projection_matrix()

    def set_pose(self, pose):
        self.__pose = pose
        self._compute_projection_matrix()

    def _compute_projection_matrix(self):
        self.__P = self.K @ self.__pose.inverse.Rt


    @property
    def K(self) -> NDArray:
        """Get the (3, 3) camera intrinsics matrix"""
        return self.__K

    @property
    def K_inv(self) -> NDArray:
        """Get the (3, 3) inverse camera intrinsics matrix"""
        return self.__K_inv

    @property
    def P(self) -> NDArray:
        """Get the (3, 4) camera projection matrix"""
        return self.__P

    @property
    def pose(self) -> Pose:
        """Get the camera pose"""
        return self.__pose


    def backproject_to_world_using_depth(self, coordinates: NDArray, depth: NDArray) -> NDArray:
        """
        Backproject 2D pixel coordinates to 3D points world coordinates using the provided depth.

        That is: "3D world points = Rt @ depth @ K^-1 @ coordinates"

        Inputs
        - coordinates:  (..., 2) or (..., 3)
        - depth:        (..., )

        Outputs
        - points_world: (..., 3)

        Attention: do not forget to reject any pixel whose depth is negative!
        """

        # Make coordinates homogeneous
        if coordinates.shape[-1] == 2: coordinates = homogenized(coordinates)

        if not coordinates.shape[:-1] == depth.shape:
            raise ValueError(f"The shape of coordinates {coordinates.shape} and depth {depth.shape} do not match. "
                             f"They should be in the forms (d_0, ..., d_N, 3) and (d_0, ..., d_N).")

        # Get normalized homogeneous pixel coordinates
        normalized_pixels = (self.K_inv @ coordinates[..., None])[..., 0] # (..., 3)

        # Project using depth
        points_camera = depth[..., None] * normalized_pixels # (..., 3)

        # Apply extrinsics
        points_world = (self.pose.Rt @ homogenized(points_camera)[..., None])[..., 0] # (..., 3)

        return points_world


    def backproject_camera_frame(self, coordinates: NDArray) -> NDArray:
        """
        TODO
        Backproject 2D homogeneous (..., 3), or image (..., 2) coordinates (with scale assumed to be 1) to camera frame 3D
        coordinates.
        """

        # Make coordinate homogeneous
        if coordinates.shape[-1] == 2:
            coordinates = homogenized(coordinates)
        assert coordinates.shape[-1] == 3

        # Compute normalized camera coordinates
        return dehomogenized((self.K_inv @ coordinates[..., np.newaxis])[..., 0]) # (N, 2)


    def project(self, points: NDArray, return_depth: bool = False) -> Union[NDArray, Tuple[NDArray, NDArray]]:
        """
        Project 3D homogeneous (..., 4) or euclidean (..., 3) points from the world frme to the image in 2D pixel
        coordinates (... , 2).

        Inputs
        - points:       (..., 4) or (..., 3) homogeneous or euclidean points given in the world frame
        - return_depth: boolean indicating whether the depth of the world points is also returned by the function or not

        Outputs
        - pixels:       (.., 2) 2D pixel coordinates of the world points reprojected on the image
        - depth:        (...,) the depth of the world points, only returned if return_depth is set to True
        """
        if points.shape[-1] == 3: points = homogenized(points)

        # Project world points
        pixels = dehomogenized((self.P @ points[..., None])[..., 0])

        # Compute depth if required
        if return_depth:
            depth = (self.pose.inverse.Rt @ points[..., None])[..., 0][..., 2] # (...,)
            return pixels, depth

        return pixels


    def backproject(self, coordinates: NDArray) -> NDArray:
        """
        Backproject 2D homogeneous (..., 3), or image (..., 2) coordinates (with scale assumed to be 1) to extrinsics frame
        3D coordinates.

        Inputs
        - coordinates   NDArray of shape (..., 3) or (..., 2) (scale assumed to be 1 in this case)

        Outputs
        - points        NDArray of shape (..., 3)
        """
        # Compute camera frame points
        points = self.backproject_camera_frame(coordinates) # (..., 3)

        # Apply extrinsics
        if self.pose != Pose():
            # p_E = T_E_C * p_C
            points = self.pose * points
        return points

    def in_front_camera_frame(self, points: NDArray) -> NDArray:
        # Check if depth is > 0
        return points[..., 2] > 0

    def in_front(self, points: NDArray) -> NDArray:
        # Apply extrinsics
        return self.in_front_camera_frame(self.transform_to_camera_frame(points))

    def project_camera_frame_homogeneous(self, points: NDArray) -> NDArray:
        """
        Project 3D Euclidean (..., 3) points given in the camera coordinate frame to homogeneous image coordinates of
        size (..., 3).
        """
        return (self.K @ points[..., np.newaxis])[..., 0]

    def project_camera_frame(self, points: NDArray) -> NDArray:
        """
        Project 3D Euclidean (..., 3) points given in the camera coordinate frame to image coordinates of size (..., 2).
        """
        return dehomogenized(self.project_camera_frame_homogeneous(points))

    def project_homogeneous(self, points: NDArray) -> NDArray:
        """
        Project 3D homogeneous (..., 4) or Euclidean (..., 3) points given in the coordinate frame described by the
        extrinsics to homogeneous image coordinates of size (..., 3).

        Inputs
        - points                    NDArray of shape (..., 3), or (..., 4) where the last row is all ones

        Outputs
        - homogeneous_coordinates   NDArray of shape (..., 3) representing homogeneous image coordinates
        """
        # Make homogeneous
        if points.shape[-1] == 3:
            points = homogenized(points)
        assert points.shape[-1] == 4

        # Project
        return (self.P @ points[..., np.newaxis])[..., 0]

    def transform_to_camera_frame(self, points: NDArray) -> NDArray:
        """
        Transform points in the extrinsic frame to the camera frame.
        """
        if self.pose != Pose():
            return self.pose.inverse * points
        return points

    def transform_to_extrinsic_frame(self, points: NDArray) -> NDArray:
        """
        Transform points in the camera frame to the extrinsic frame.
        """
        if self.pose != Pose():
            return self.pose * points
        return self.pose * points