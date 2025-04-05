from __future__ import annotations

# NumPy
import numpy as np
from numpy.typing import NDArray

# Python
from typing import Tuple, Union, List
from pathlib import Path

# ROS
try:
    import rospy
    from std_msgs.msg import Header, ColorRGBA
    from geometry_msgs.msg import Vector3
    from visualization_msgs.msg import Marker
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Matplotlib
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Utils
from sba.utils.base import normalized
from sba.utils.pose import Pose
from sba.utils.cameras import PinholeCamera

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)


class Cylinder:

    def __init__(self, pose: Pose = Pose(), radius: float = 1, height: float = 1):
        """
        Creates a cylinder instance given its pose, radius and height.

        Inputs
        - pose:   Pose of the cylinder. The pose defines the bottom center fo the cylinder and the normal vector given
                  by the z-axis of the pose.
        - radius: radius of the cylinder
        - height: height of the cylinder
        """

        assert radius >= 0 and "Radius can't be negative."
        assert height >= 0 and "Height can't be negative."

        self.__pose = pose
        self.__radius = radius
        self.__height = height

    @staticmethod
    def from_center_and_normal(self, center: NDArray, normal: NDArray, radius: float, height: float) -> Cylinder:
        """
        Creates a cylinder instance given its bottom center, normal vector, radius and height.

        Inputs
        - center: 3D position of the bottom center of the cylinder
        - normal: normal vector of the cylinder
        - radius: radius of the cylinder
        - height: height of the cylinder
        """

        assert normal.shape == (3,) and center.shape == (3,)

        n = normalized(normal)
        u = np.cross(n, [1, 0, 0])
        if np.linalg.norm(u) < 1e-8:
            u = np.cross(n, [0, 1, 0])
        u = normalized(u)
        v = -np.cross(u, n)
        R = np.array([u, v, n]).T

        return Cylinder(Pose(R, center), radius, height)

    @property
    def pose(self) -> Pose:
        return self.__pose

    @property
    def radius(self) -> float:
        return self.__radius

    def set_radius(self, radius: float) -> float:
        assert radius >= 0 and "Radius can't be negative."
        self.__radius = radius

    @property
    def height(self) -> float:
        return self.__height

    def set_height(self, height: float) -> float:
        assert height >= 0 and "Height can't be negative."
        self.__height = height

    @property
    def normal(self) -> NDArray:
        return self.pose.R[:, 2]

    @property
    def center(self) -> NDArray:
        return self.pose.t

    @property
    def lower_pose(self) -> NDArray:
        return self.pose

    @property
    def middle_pose(self) -> NDArray:
        return self.pose * Pose(t = np.array([0, 0, self.height/2]))

    @property
    def upper_pose(self) -> NDArray:
        return self.pose * Pose(t = np.array([0, 0, self.height]))

    @property
    def upper_center(self) -> NDArray:
        return self.upper_pose.t

    def to_string(self) -> NDArray:
        q = self.pose.quat_wxyz
        return f"q {q[0]} {q[1]} {q[2]} {q[3]} " + \
               f"t {self.pose.t[0]} {self.pose.t[1]} {self.pose.t[2]} " + \
               f"r {self.radius} h {self.height}"

    @staticmethod
    def from_string(text: str) -> Cylinder:
        parts = text.split()
        if len(parts) != 13:
            raise ValueError(f"Given string '{text}' does not contain 13 distinct parts.")
        if parts[0] != "q":
            raise ValueError(f"Given string '{text}' does not start with 'q'.")
        if parts[5] != "t":
            raise ValueError(f"Given string '{text}' does not have 't' at the right position.")
        if parts[9] != "r":
            raise ValueError(f"Given string '{text}' does not have 'r' at the right position.")
        if parts[11] != "h":
            raise ValueError(f"Given string '{text}' does not have 'h' at the right position.")

        q = np.array([float(parts[i]) for i in range(1, 1+4)])
        t = np.array([float(parts[i]) for i in range(6, 6+3)])
        r = float(parts[10])
        h = float(parts[12])

        return Cylinder(Pose.from_quat_wxyz(q, t), r, h)

    @staticmethod
    def from_text_file(file_path: Union[str, Path]) -> List[Cylinder]:
        cylinders = []
        file_path = Path(file_path)
        if not file_path.is_file():
            raise FileNotFoundError(f"The provided file path '{file_path}' does not exist.")
        with open(file_path, 'r') as file:
            for line in file:
                try:
                   cylinders.append(Cylinder.from_string(line.strip()))
                except ValueError:
                    raise ValueError(f"Error in creating cylinder from line 'line.strip()'.")
        return cylinders


    # ROS methods
    if ROS_AVAILABLE:

        def to_ros_marker(self, header: Header = Header(seq=0, stamp=None, frame_id="map"), ns: str = None,
                          id: int = None, frame_locked: bool = True, color: Union[NDArray, ColorRGBA] = ColorRGBA(1, 0, 0, 1)) -> Marker:

            # Create a marker
            marker = Marker()
            if header is not None:
               marker.header = header
            if ns is not None:
                marker.ns = ns
            if id is not None:
                marker.id = id

            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.lifetime = rospy.rostime.Duration(0)
            marker.frame_locked = frame_locked

            # position, orientation and scale
            marker.pose = self.middle_pose.to_ros_pose()
            marker.scale = Vector3(self.radius*2, self.radius*2, self.height)

            # color
            if type(color) == np.ndarray:
                color = ColorRGBA(*color)
            marker.color = color

            return marker


    def get_3d_points(self, num: int = 300) -> NDArray:
        thetas = np.linspace(0, 2 * np.pi, num=num)

        u = self.pose.R[:, 0] * self.radius
        v = self.pose.R[:, 1] * self.radius

        points = [self.center + np.cos(theta) * u + \
                   np.sin(theta) * v for theta in thetas]
        points_upper = [self.upper_center + np.cos(theta) * u + \
                         np.sin(theta) * v for theta in thetas]
        points.extend(points_upper)
        return np.array(points)


    def visualize(self, num: int = 300) -> Axes3D:
        # Create figure
        fig = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d')

        points = self.get_3d_points(num = num)
        ax.plot(points[:num, 0], points[:num, 1], points[:num, 2], c='c')
        ax.plot(points[num:, 0], points[num:, 1], points[num:, 2], c='m')

        points = np.array([self.center, self.center + self.normal * self.height])
        ax.plot(points[:, 0], points[:, 1], points[:, 2], c='k')

#        ax.axis('equal')

        self.pose.show(ax, scale=self.radius/10)

        self.upper_pose.show(ax, scale=self.radius/10)

        # Title
        ax.set_title('Cylinder visualization')

        # Axis labels
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        return ax

    @staticmethod
    def project_circle(pose: Pose, radius: float, camera: PinholeCamera) \
          -> NDArray:

        # Transformation from circle plane pi to camera frame
        pose_world_pi = pose
        pose_camera_pi = camera.pose.inverse * pose_world_pi

        # Compute homography matrix H
        T = np.c_[pose_camera_pi.R[:, :2], pose_camera_pi.t[:, None]] # (3,3)
        H = camera.K @ T # (3,3)

        # Assert it is invertible
        assert abs(np.linalg.det(H)) > 1e-4
        H_inv = np.linalg.inv(H)

        # Build the homogenous circle matrix
        C_3D = np.zeros((3,3))
        C_3D[0, 0] = 1/radius**2
        C_3D[1, 1] = 1/radius**2
        C_3D[2, 2] = -1

        # Compute the projected circle homogeneous matrix
        C_2D = H_inv.T @ C_3D @ H_inv
        C_2D /= -C_2D[2,2] # up-to-scale -> have -1 at (2,2)

        assert np.all(np.abs(C_2D - C_2D.T) < 1e-8) # symmetric

        return C_2D

    def project_circles(self, camera: PinholeCamera) -> Tuple[NDArray, NDArray]:

        return Cylinder.project_circle(self.pose, self.radius, camera), \
               Cylinder.project_circle(self.upper_pose, self.radius, camera)

    def project_circles_to_points(self, camera: PinholeCamera) \
          -> Tuple[NDArray, NDArray]:
        C1, C2 = self.project_circles(camera)
        return plot_ellipse(C1), plot_ellipse(C2)

    def project_edge_points(self, camera: PinholeCamera) -> NDArray:
        edge_points = self.get_edge_points(camera)
        return camera.project(edge_points)

    def get_edge_points(self, camera: PinholeCamera) -> NDArray:

        # Project the camera center into the plane
        camera_projected_2d = self.pose.inverse * camera.pose.t # (3,)
        camera_projected_2d[2] = 0 # (zero z-component)

        # Circle center to camera center vector
        d = camera_projected_2d - np.array([0, 0, 0]) # (3,)
        unit_d = d / np.linalg.norm(d)

        # Compute the angle
        if np.linalg.norm(d) <= self.radius:
            raise ValueError("Camera center is inside the infinite cylinder.")
        beta = np.arccos(self.radius / np.linalg.norm(d))

        # Compute interesecting points
        rot_pos = Pose.from_rotation_angle_and_axis(beta, np.array([0, 0, 1]))
        rot_neg = Pose.from_rotation_angle_and_axis(-beta, np.array([0, 0, 1]))

        p1 = rot_pos * unit_d * self.radius
        p2 = rot_pos * unit_d * self.radius + np.array([0, 0, self.height])
        p3 = rot_neg * unit_d * self.radius + np.array([0, 0, self.height])
        p4 = rot_neg * unit_d * self.radius

        edge_points_w_plane = np.array([p1, p2, p3, p4]) # (4, 3)
        edge_points_w_world = self.pose * edge_points_w_plane

        return edge_points_w_world

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.to_string()

def plot_ellipse(C: NDArray, num: int = 100) -> NDArray:
    """
    Ellipse described by homogeneous equation x^T C x = 0, with x = (x, y, 1).
    For a symmetric matrix C and properties to be a proper ellipse (i.e. not a
    hyperbola).

    Equivalent to:
    1/2 x^T A x + b^T x + c = 0 with x = (x, y)
    1/2 (x - x_0)^T A (x-x_0) + c' = 0 with x = (x, y)
	where A must be positive definite if c' < 0
       or A must be negative definite if c' > 0
    """
    assert C.shape == (3,3)
    assert np.all(np.abs(C - C.T) < 1e-8)

    if C[2,2] == 0:
        print("Zero at C(2,2) -> No points")
        return np.array([])

    # Up-to-scale
    C = C / C[2, 2] # In order to have +1 at (2,2)

    # 1/2 * x^T A x + b^T x + c = 0
    A = 2 * C[:2, :2] # (2, 2), symmetric, positive definite
    b = 2 * C[:2, 2] # (2,)
    c = C[2, 2] #TODO should be -1 -> = 1

    if abs(np.linalg.det(A)) < 1e-16:
        logger.warn("Not invertible matrix A. Returning empty array.")
        return np.array([])

    x_0 = np.linalg.solve(A, -b)
    c = c - 0.5 * x_0[None, :] @ A @ x_0

    if np.all(np.linalg.eigvals(A) < 0):
       A *= -1
       c *= -1
       print("Neg def -> pos def")

    if not (np.all(np.linalg.eigvals(A) > 0) and c < 0):
        logger.warn("Not a proper ellipse. Returning empty array.")
        return np.array([])

    lambdas, eigvecs = np.linalg.eig(A)

    thetas = np.linspace(0, 2 * np.pi, num = num)

    points = [x_0 + \
              np.cos(theta) * np.sqrt(-2 * c / lambdas[0]) * eigvecs[:, 0] + \
              np.sin(theta) * np.sqrt(-2 * c / lambdas[1]) * eigvecs[:, 1] \
               for theta in thetas]

    points = np.array(points)
    return points