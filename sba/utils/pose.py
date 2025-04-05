from __future__ import annotations

# NumPy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import Tuple, Union, List, Optional

# ROS
try:
    import geometry_msgs.msg
    import std_msgs.msg
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# COLMAP
try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False

# Matplotlib
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# SciPy
from scipy.spatial.transform import Rotation

# Utils
from .base import homogenized

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

class Pose:

    # Default constructor
    def __init__(self, R: NDArray = np.eye(3),
                 t: Union[NDArray, List] = np.zeros(3),
                 tol: float = 1e-12) -> Pose:
        """Constructor of the Pose class"""
        self.set_t(t)
        self.set_R(R, tol)

    # Setters
    def set_R(self, R: NDArray, tol: float = 1e-12):
        """
        Set the rotation matrix R

        Requirements:
        - 3x3 real matrix
        - rotation matrix, i.e. R^T @ R = I
        - right handed, i.e. det(R) = +1
        """
        assert R.shape == (3,3)
        assert np.abs(np.linalg.det(R) - 1) < tol
        assert np.all(np.abs((R.T @ R - np.eye(3))) < tol)
        self.__R = R
        self.__inverse = Pose._compute_inverse(self)
        self.__quat_wxyz = Rotation.from_matrix(self.R).as_quat()[[3, 0, 1, 2]]


    def set_t(self, t: Union[NDArray, List]):
        """Set the translation vector t"""
        t = np.squeeze(np.array(t)).astype(float)
        assert t.shape == (3,)
        self.__t = t # t : 1D array (3,)
        if hasattr(self, '__R'):
            self.__inverse = Pose._compute_inverse(self)

    # Inverse computation
    @classmethod
    def _compute_inverse(cls, pose: Pose) -> Pose:
        """Class method that computes and sets the inverse of a Pose."""

        # Create a new instance of the class without calling the constructor
        inv = cls.__new__(cls)
        inv.__R = pose.R.T
        inv.__t = -pose.R.T @ pose.t
        inv.__inverse = pose
        inv.__quat_wxyz = Rotation.from_matrix(pose.R.T).as_quat()[[3, 0, 1, 2]]

        # Prohibit editing the attributes of the inverse Pose
        inv.set_R = cls._inverse_readonly_error
        inv.set_t = cls._inverse_readonly_error
        return inv


    # Inverse read-only error
    @staticmethod
    def _inverse_readonly_error():
        logger.error(
            "Can't set the rotation matrix or the translation vector of the " +
            "inverse of the Pose instace.")
        raise AttributeError(
            "Can't set the rotation matrix or the translation vector of the " +
            "inverse of the Pose instace.")

    # Copy
    def copy(self) -> Pose:
        return Pose(self.R.copy(), self.t.copy())


    # Properties
    @property
    def R(self) -> NDArray:
        return self.__R


    @property
    def t(self) -> NDArray:
        return self.__t


    @property
    def Rt(self) -> NDArray:
        """Get the 3x4 transformation matrix"""
        return np.c_[self.R, self.t]


    @property
    def matrix(self) -> NDArray:
        """Get the 4x4 transformation matrix"""
        return np.r_[self.Rt, np.array([[0, 0, 0, 1]])]


    @property
    def inverse(self) -> Pose:
        return self.__inverse


    @property
    def quat_xyzw(self) -> NDArray:
        """Get the quaternion representation of the rotation, as (x, y, z, w)"""
        q = self.__quat_wxyz[[1,2,3,0]]
        q2 = Rotation.from_matrix(self.R).as_quat()
        assert np.all(np.abs(q - q2) < 1e-8)
        return self.__quat_wxyz[[1,2,3,0]]


    @property
    def quat_wxyz(self) -> NDArray:
        """Get the quaternion representation of the rotation, as (w, x, y, z)"""
        q = self.__quat_wxyz
        q2 = Rotation.from_matrix(self.R).as_quat()[[3, 0, 1, 2]]
        assert np.all(np.abs(q - q2) < 1e-8)
        return self.__quat_wxyz


    # Methods
    def angle_axis(self) -> NDArray:
        return Rotation.from_matrix(self.R).as_rotvec()


    def rotation_angle_and_axis(self) -> Tuple[float, NDArray]:
        aa = self.angle_axis()
        angle = np.linalg.norm(aa)
        axis = aa / angle
        return angle, axis


    def distance(self) -> float:
        return np.linalg.norm(self.t)


    # Pose errors
    @staticmethod
    def distance_error(p1: Pose, p2: Pose) -> float:
        return np.linalg.norm(p1.t - p2.t)


    @staticmethod
    def angular_error(p1: Pose, p2: Pose, degrees: bool = False) -> float:
        pose_diff = p1 * p2.inverse
        angle = np.linalg.norm(pose_diff.angle_axis())
        return angle if not degrees else np.rad2deg(angle)


    @staticmethod
    def error(p1: Pose, p2: Pose, degrees: bool = False) -> Tuple[float, float]:
        return Pose.distance_error(p1, p2), \
               Pose.angular_error(p1, p2, degrees=degrees)


    # Additional constructors
    @staticmethod
    def random(t_bound: float = 1) -> Pose:
        """
        Generates a random Pose with any orientation and position in the bounded
        cube [-t_bound, +t_bound]^3.
        """
        t = (np.random.random(3) - 0.5) * 2 * t_bound
        zyx = np.random.random(3) * 2 * np.pi # radians
        r = Rotation.from_euler('zyx', zyx)
        return Pose(r.as_matrix(), t)


    @staticmethod
    def from_rotation_angle_and_axis(angle: float, axis: NDArray,
                                     t: NDArray = np.zeros(3)) -> Pose:
        """
        Get a Pose from a angle in RAD and a 3D axis vector.

        Note: The axis vector does not necessarily need to be normalized.
        """
        axis = np.squeeze(axis).astype(float)
        assert axis.shape == (3,)
        axis = axis / np.linalg.norm(axis) # normalize
        rot_vec = axis * angle
        return Pose.from_rotation_vector(rot_vec, t)


    @staticmethod
    def from_rotation_vector(rot_vec: NDArray,
                             t: NDArray = np.zeros(3)) -> Pose:
        """
        Get a Pose from a rotation vector that captures the axis and the angle
        through its norm.
        """
        R = Rotation.from_rotvec(rot_vec).as_matrix()
        return Pose(R, t)


    @staticmethod
    def from_quat_wxyz(q: NDArray, t: NDArray = np.zeros(3),
                       tol: float = 1e-12) -> Pose:
        """
        Get a Pose from quaternions q = (w, x, y, z) and a translation vector
        t = (x, y, z). The quaternion will be normalized.
        """
        q = q.squeeze()
        assert q.shape == (4,)
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() # function assumes quaternion (x, y, z, w)
        return Pose(R, t)


    @staticmethod
    def from_quat_xyzw(q: NDArray, t: NDArray = np.zeros(3), tol: float = 1e-12) -> Pose:
        """ Get a Pose from quaternions q = (x, y, z, w) and a translation vector t = (x, y, z)
            The quaternion will be normalized. """
        q = q.squeeze()
        assert q.shape == (4,)
        return Pose.from_quat_wxyz(q[[3,0,1,2]], t, tol)


    @staticmethod
    def from_matrix(matrix: NDArray, tol: float = 1e-12) -> Pose:
        """ Get a Pose from a 4x4 homogeneous matrix """
        assert matrix.shape == (4,4) and "Not a 4x4 homogeneous matrix"
        return Pose(matrix[:3, :3], matrix[:3, 3], tol=tol)


    # ROS constructors and methods
    if ROS_AVAILABLE:
        @staticmethod
        def from_ros_pose(pose_msg: geometry_msgs.msg.Pose) -> Pose:
            """Get a Pose from a ROS pose"""
            q = np.array([pose_msg.orientation.w, pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z])
            t = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])
            return Pose.from_quat_wxyz(q, t)


        @staticmethod
        def from_ros_transform(transform_msg: geometry_msgs.msg.Transform) -> Pose:
            """Get a Pose from a ROS pose"""
            q = np.array([transform_msg.rotation.w, transform_msg.rotation.x, transform_msg.rotation.y, transform_msg.rotation.z])
            t = np.array([transform_msg.translation.x, transform_msg.translation.y, transform_msg.translation.z])
            return Pose.from_quat_wxyz(q, t)


        def to_ros_pose(self) -> geometry_msgs.msg.Pose:
            """Transform the Pose into a ROS pose"""
            return geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(*self.t),
                                      orientation=geometry_msgs.msg.Quaternion(*self.quat_xyzw))


        def to_ros_pose_stamped(self, header: std_msgs.msg.Header) -> geometry_msgs.msg.PoseStamped:
            return geometry_msgs.msg.PoseStamped(header, self.to_ros_pose())


        def to_ros_transform(self) -> geometry_msgs.msg.Transform:
            return geometry_msgs.msg.Transform(translation=geometry_msgs.msg.Vector3(*self.t),
                                           rotation=geometry_msgs.msg.Quaternion(*self.quat_xyzw))


        def to_ros_transform_stamped(self, header: std_msgs.msg.Header, child_frame_id: str) \
            -> geometry_msgs.msg.TransformStamped:
            return geometry_msgs.msg.TransformStamped(header, child_frame_id, self.to_ros_transform())


    # COLMAP constructors and methods
    if COLMAP_AVAILABLE:
        @staticmethod
        def from_colmap_image(image: pycolmap.Image, include_name: bool = True) -> Pose:
            """
            Get a Pose from a colmap image.
            Note: It returns the transformation from camera to world, that is t = (tx, ty, tz) contains the coordinates of
              the camera in the world.
            """
            pose = Pose.from_quat_xyzw(image.cam_from_world.rotation.quat,
                                       image.cam_from_world.translation).inverse
            if include_name: pose.name = image.name
            return pose


        @staticmethod
        def set_colmap_image_pose(image: pycolmap.Image, pose:Pose) -> None:
            """
            Sets the pose of the colmap image to the given Pose.
            Note: The given Pose must be the transformation from camera to world, that is t = (tx, ty, tz) contains the
                  coordinates of the camera in the world.
            """
            q = pose.inverse.quat_xyzw
            t = pose.inverse.t
            image.cam_from_world.rotation.quat = q
            image.cam_from_world.translation = t
            return


    # Multiplication operator overload
    def __mul__(self, x: Union[NDArray, Pose]):
        """
        Multiplication '*' operator overload.

        With a NDArray: y = pose * x
        - x: NDArray of shape (..., 3) or (..., 3, 1) representing 3D vectors
        - y: NDArray of shape (..., 3) or (..., 3, 1) representing the 3D vectors after the transformation described by the Pose is applied

        With another Pose: pose_A_C = pose_A_B * pose_B_C.
        - pose_B_C:  another instance of the Pose class
        - pose_A_C: a Pose representing the consecutive transformation of the two transformations given by pose_A_B and pose_B_C
        """

        # With a NDArray: y = pose * x
        if isinstance(x, np.ndarray):
            # x: (..., 3) or (..., 3, 1)
            if x.shape[-1] == 1:
                x = x[..., 0]
            assert x.shape[-1] == 3
            return (self.Rt @ homogenized(x)[..., np.newaxis])[..., 0]

        # With another Pose: pose_A_C = pose_A_B * pose_B_C.
        elif isinstance(x, Pose):
            pose_matrix = self.matrix @ x.matrix # (4,4)
            return Pose(pose_matrix[:3, :3], pose_matrix[:3, 3])

        # Undefined multiplication
        else:
            raise NotImplementedError(f"Multiplication of a Pose with {type(x)} is not defined.")


    def __repr__(self):
        return self.__str__()


    def __str__(self):
        return "Pose with rotation R:\n" + str(np.round(self.__R,2)) + \
               "\ntranslation t:\n " + str(np.round(self.__t,2))

    # Visualization functions
    if MATPLOTLIB_AVAILABLE :
        @staticmethod
        def _plot_arrow(ax: Axes3D, origin: NDArray, vector: NDArray, **args) -> None:
            ax.plot3D([origin[0], origin[0] + vector[0]],
                      [origin[1], origin[1] + vector[1]],
                      [origin[2], origin[2] + vector[2]],
                      color="k")

            # Plot the arrows
            ax.quiver(origin[0], origin[1], origin[2],
                      vector[0], vector[1], vector[2],
                      **args)
            return

        def show(self, axes: Optional[Axes3D] = None, scale: Optional[float] = None) -> Axes3D:
            """
            Visualize the Pose in a 3D matloptlib plot.
            """
            Pose.visualize(self, axes, scale)

        @staticmethod
        def visualize(poses: Union[Pose, List[Pose]], axes: Optional[Axes3D] = None, scale: Optional[float] = None) -> Axes3D:
            """
            Visualize a list of Poses in a 3D matloptlib plot.

            Inputs:
            - poses: list of Poses to plot
            """

            if type(poses) == Pose:
                poses = [poses]

            # No axes provided
            if axes is None:
                # Create figure
                fig = plt.figure()
                ax: Axes3D = fig.add_subplot(111, projection='3d')

                # Title
                ax.set_title('Poses visualization' if len(poses) > 1 else 'Pose visualization')

                # Axis labels
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')

            # Axes provided
            else:
                ax = axes

            # Access poses origins
            origins = np.array([pose.t for pose in poses]) # (N, 3)
            origins_max = np.max(origins, axis = 0)
            origins_min = np.min(origins, axis = 0)
            max_diff = np.max(origins_max - origins_min)
            max_diff = max(max_diff, 0.1)

            # Plot poses
            for i, pose in enumerate(poses):

                # Origin
                ax.scatter(*list(pose.t), c='k', marker='o')

                # Direction vectors
                s = scale if scale is not None else max_diff/15 # scale of the arrows

                params = {'arrow_length_ratio': 0.3, 'linewidth': 2}
                Pose._plot_arrow(ax, pose.t, s * pose.R[:, 0], color='r', **params)
                Pose._plot_arrow(ax, pose.t, s * pose.R[:, 1], color='g', **params)
                Pose._plot_arrow(ax, pose.t, s * pose.R[:, 2], color='b', **params)

                # Text
                if not (len(poses) == 1 and not hasattr(pose, 'name')):
                    text = i if not hasattr(pose, 'name') else pose.name
                    ax.text(*list(pose.t), text, color='k', fontsize=12)

            if axes is None:
                # Equal axis
#                ax.axis('equal')
                ax.set_box_aspect([1, 1, 1])

                # Show
                plt.show()

            return ax
