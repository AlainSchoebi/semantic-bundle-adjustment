import numpy as np

from .base import homogenized
from .pose import Pose

# Logging
from loggers import get_logger
import logging
logger = get_logger(__name__)

def eq(a, b):
    if type(a) == Pose and type(b) == Pose:
        a, b = a.matrix, b.matrix
    assert np.all(np.abs(a - b) < 1e-8)

if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)

    q = np.random.randn(4)
    q /= np.linalg.norm(q)

    t = np.random.randn(3)

    # Default tests
    pose = Pose.from_quat_wxyz(q, t)

    eq(pose.R @ pose.R.T, np.eye(3))
    eq(pose.R @ pose.inverse.R, np.eye(3))

    x = np.random.randn(3)

    eq(x, pose.inverse.Rt @ homogenized((pose.Rt @ homogenized(x))))
    eq(x, pose.inverse * (pose * x))

    eq(pose.quat_wxyz[[1, 2, 3, 0]], pose.quat_xyzw)
    eq(pose.inverse.quat_wxyz[[1, 2, 3, 0]], pose.inverse.quat_xyzw)

    eq(Pose.from_quat_wxyz(pose.quat_wxyz *
    np.array([-1, 1, 1, 1])).R, pose.inverse.R)
    eq(Pose.from_quat_xyzw(pose.quat_xyzw *
    np.array([1, 1, 1, -1])).R, pose.inverse.R)

    logger.info(f"Successfully passed all the default tests.")

    # ROS tests
    try:
        import geometry_msgs.msg
        import std_msgs.msg
        ROS_AVAILABLE = True
    except ImportError:
        ROS_AVAILABLE = False

    if ROS_AVAILABLE:
        eq(Pose.from_ros_pose(pose.to_ros_pose()).matrix, pose.matrix)

        logger.info(f"Successfully passed all the ROS tests.")

    # COLMAP tests
    try:
        import pycolmap
        COLMAP_AVAILABLE = True
    except ImportError:
        COLMAP_AVAILABLE = False

    if COLMAP_AVAILABLE:

        image = pycolmap.Image()
        Pose.set_colmap_image_pose(image, pose)
        eq(pose, Pose.from_colmap_image(image))

        # pose: camera -> world
        # image: world -> camera
        eq(image.qvec, pose.inverse.quat_wxyz)

        logger.info(f"Successfully passed all the COLMAP tests.")