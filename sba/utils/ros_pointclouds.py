# Numpy
import numpy as np
from numpy.typing import NDArray

# Python
from tqdm import tqdm

# ROS
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs.point_cloud2 import read_points
from std_msgs.msg import Header
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker, MarkerArray

# Utils
from sba.utils.pose import Pose

def points_to_pointcloud(points: NDArray, header: Header = Header(seq=0, stamp=None, frame_id="map")) -> PointCloud2:
    """
    Transform an array of 3D points to a ROS PointCloud2 message with a given header.
    """
    if not (np.ndim(points) == 2 and points.shape[1] == 3):
        raise ValueError("points should have been (N, 3) but was " + str(points.shape))

    # message
    cloud_msg = PointCloud2()
    cloud_msg.header.frame_id = header.frame_id
    cloud_msg.header.stamp = header.stamp

    # data
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    cloud_msg.data = points.astype(dtype).tobytes()

    # shape
    cloud_msg.height = 1 # unordered list
    cloud_msg.width = points.shape[0]
    cloud_msg.point_step = itemsize * points.shape[1]
    cloud_msg.row_step = itemsize * points.shape[1] * points.shape[0]

    # fields
    cloud_msg.fields = [
        PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate("xyz")
    ]

    cloud_msg.is_bigendian = False
    cloud_msg.is_dense = False

    return cloud_msg


def points_to_marker(points: NDArray, colors: NDArray, scale: int = 0.2,
                     header: Header = Header(seq=0, stamp=None, frame_id="map"),
                     id: int = None, frame_locked: bool = True) -> Marker:
    """
    Creates a ROS Marker message containing all the points displayed as spheres. Each point is colored by its
    corresponding color in the colors array. The marker is of type "SPHERE_LIST".

    Inputs
    - points:       (N, 3) array containing the N 3D points to display.
    - colors:       (N, 4) array containing the N RGBA corresponding point colors.
    - scale:        scale of each sphere in the visualization in meters.
    - header:       header of the marker message.
    - id:           id of the marker. Note that only one marker per id is displayed in RViz, that is they must be unique.
    - frame_locked: f this marker should be frame-locked, i.e. retransformed into its frame every timestep. ???

    Returns
    - marker:       the marker message containing the colored pointcloud
    """

    if not (np.ndim(points) == 2 and points.shape[1] == 3):
        raise ValueError(f"The given points should have shape (N, 3), but was {points.shape}.")

    if not (np.ndim(colors) == 2 and colors.shape[1] == 4):
        raise ValueError(f"The given colors should have shape (N, 4), but was {colors.shape}.")

    if not len(colors) == len(points):
        raise ValueError(f"The number of points ({len(points)}) and colors ({len(colors)}) are different.")

    # Create a marker
    marker = Marker()
    if header is not None:
       marker.header = header

    if id is not None:
        marker.id = id

    marker.type = Marker.SPHERE_LIST
    marker.action = Marker.ADD
    marker.lifetime = rospy.rostime.Duration(0)
    marker.frame_locked = frame_locked

    marker.pose = Pose().to_ros_pose()
    marker.scale = Vector3(scale,scale,scale)

    marker.points = [Point(*point) for point in tqdm(points, leave=False, desc="Copying points")]
    marker.colors = [ColorRGBA(*color) for color in tqdm(colors, leave=False, desc="Copying colors")]

    return marker


def pointcloud_to_points(msg: PointCloud2) -> NDArray:
    """
    Transform a ROS PointCloud2 message of 3D points into a NDArray.
    """

    # Get point cloud data
    data = list(read_points(msg, field_names=("x", "y", "z"), skip_nans=False))

    # Convert to numpy array
    points = np.array(data, dtype=np.float32)

    # Reshape the array to the correct shape (N, 3)
    points = points.reshape(-1, 3)

    return points