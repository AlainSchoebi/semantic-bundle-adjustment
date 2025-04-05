# Typing
from __future__ import annotations

# Python
from tqdm import tqdm

# Numpy
import numpy as np
from numpy.typing import NDArray

# OpenCV
import cv2

# ROS
import rospy
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point, Vector3
from visualization_msgs.msg import Marker

# Utils
from .base import homogenized
from .pose import Pose
from .cameras import PinholeCamera


def image_with_camera_pose_to_marker(image: NDArray, camera: PinholeCamera, focal_length: float = 1,
                                     max_length_pixels: int = None,
                                     header: Header = Header(seq=0, stamp=None, frame_id="map"), ns: str = None,
                                     id: int = None, frame_locked: bool = True, ) -> Marker:
    """
    Creates a marker from a RGB or RGBA image, given a camera pose (camera to world), the intrinsics and the desired
    focal length [m] at which the image should be displayed.

    TODO
    max_length_pixels
    """

    if abs(camera.K[0,0] - camera.K[1,1]) > 1e-4:
        raise NotImplementedError(f"Cannot handle not square pixels. K is {camera.K}.")

    # Extra info from the intrinsics K
    alpha = camera.K[0,0]             # scale [pixels/meter]
    principal_point = camera.K[:2, 2] # principal point [pixels]

    # Compute the new scale and pose for the image corner
    scale = 1 / alpha * focal_length
    pose = camera.pose * Pose(t = focal_length * np.array([*-principal_point/alpha, 1]))

    # Return marker
    return image_to_marker(image, pose, scale, max_length_pixels=max_length_pixels,
                           header=header, ns=ns, id=id, frame_locked=frame_locked, )



def image_to_marker(image: NDArray, pose: Pose, scale: float,
                    header: Header = Header(seq=0, stamp=None, frame_id="map"), ns: str = None,
                    id: int = None, frame_locked: bool = True, max_length_pixels: int = None) -> Marker:
    """
    Creates a marker from a RGB or RGBA image.
    Note: the header, the id and the namespace can also be set later.
    """
    # TODO: half a pixel off...

    # Rescale if we exceed the max_length_pixels on either side
    if max_length_pixels is not None:
        h_ratio = float(max_length_pixels) / float(image.shape[0])
        w_ratio = float(max_length_pixels) / float(image.shape[1])
        ratio = min(h_ratio, w_ratio)
        if ratio < 1.0:
            new_h = int(image.shape[0] * ratio)
            new_w = int(image.shape[1] * ratio)
            resized_image = cv2.resize(image, (new_w, new_h), cv2.INTER_NEAREST)
            image = resized_image
            scale = scale / ratio

    assert image.ndim == 3
    h, w, n_channels = image.shape

    if image.dtype == np.uint8:
        image = image.astype(float)/255 # convert image to float in range [0,1]
    assert image.dtype == float and np.max(image) <= 1 and " Image should be float in range [0, 1]"

    assert n_channels == 3 or n_channels == 4 and "Not RGB nor RGBA image"
    # Turn into RGBA if RGB
    if n_channels == 3:
        image = homogenized(image)
        h, w, n_channels = image.shape

    # Create a marker
    marker = Marker()
    if header is not None:
       marker.header = header
    if ns is not None:
        marker.ns = ns
    if id is not None:
        marker.id = id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.lifetime = rospy.rostime.Duration(0)
    marker.frame_locked = frame_locked

    # position, orientation and scale
    marker.pose = pose.to_ros_pose()
    marker.scale = Vector3(*([scale]*3))

    # pixels
    pixels = np.zeros((h-1, w-1, 3), int) # (h-1, w-1, 2)
    pixels[:, :, 0] = np.arange(w-1)[np.newaxis, :] # fill u
    pixels[:, :, 1] = np.arange(h-1)[:, np.newaxis] # fill v
    pixels[:, :, 2] = 0 # fill z = 0

    # corners pixels in counter-clock-wise order (with v-axis pointing down)
    corner_pixels = np.zeros((h-1, w-1, 4, 3), int) # (h-1, w-1, 4, 3)
    corner_pixels[..., 0, :] = pixels
    corner_pixels[..., 1, :] = pixels + np.array([0, 1, 0])
    corner_pixels[..., 2, :] = pixels + np.array([1, 1, 0])
    corner_pixels[..., 3, :] = pixels + np.array([1, 0, 0])

    # access color
    corner_colors = image[corner_pixels[..., 1], corner_pixels[..., 0], :] # (h-1, w-1, 4, 4)

    # rviz cannot display semi-transparent triangles !
    corner_colors[..., 3] = 1.0

    # turn everything into triangle order
    triangle_pixels = corner_pixels[:, :, [0, 1, 2, 0, 2, 3], :] # (h-1, w-1, 6, 3)
    triangle_pixels = triangle_pixels.reshape(-1, 3) # (?, 3)

    triangle_colors = corner_colors[:, :, [0, 1, 2, 0, 2, 3], :] # (h-1, w-1, 4, 4)
    triangle_colors = triangle_colors.reshape(-1, 4) # (?, 4)

    # add points and colors to the marker
    marker.points = [Point(*triangle_pixels[i]) for i in tqdm(range(len(triangle_pixels)),
                                                              leave=False, desc="Copying points")]
    marker.colors = [ColorRGBA(*triangle_colors[i]) for i in tqdm(range(len(triangle_colors)),
                                                                  leave=False, desc="Copying colours")]
    assert len(marker.points) == len(marker.colors)

    return marker