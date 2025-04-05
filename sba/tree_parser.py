# Python
import argparse
from pathlib import Path

# Logging
from sba.utils.loggers import get_logger
import logging
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

class TreeParser(argparse.ArgumentParser):

    # Constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.added_workspace_arg_automatically = False


    # Arguments
    def _add_workspace_folder_argument(self):
        self.add_argument("--workspace_folder", type=str, required=False, default=None,
                         help='The workspace folder must be a "result/" folder from the data generation.' +
                               'That is, it must contain an "output/" folder etc.')

    def add_camera_poses_arguments(self):
        self.add_argument(
            "--camera_poses_path", type=str, required=False, default=None,
            help = "Input file or folder containing the camera poses."
            )
        self.add_argument(
            "--camera_poses_format", type=str, required=False, default=None,
            choices=["vulkan_text", "colmap_text", "colmap_model"],
            help = "'vulkan_text': provide a .txt file generaed with Vulkan. " +
                   "'colmap_text': provide  a .txt file exported form COLMAP. " +
                   "'colmap_model': provide a folder model folder form COLMAP."
            )

    def add_h5_folder_argument(self):
        self.add_argument("--h5_folder", type=str, required=False, default=None,
                          help = "Folder containing the corresponding .h5 files.")

    def add_camera_intrinsics_argument(self):
        self.add_argument(
            "--camera_intrinsics", type=float, required=False, nargs=4, default=[2559, 2559, 1536, 1152],
            help = "Camera intrinsics for a simple pinhole camera in the following order: fx, fy, cx, fy."
            )

    def add_workspace_and_camera_h5_folder_arguments(self):
        self._add_workspace_folder_argument()
        self.add_camera_poses_arguments()
        self.add_h5_folder_argument()
        self.added_workspace_arg_automatically = True


    # Parse arguments
    def parse_args(self, *args, **kwargs):
        # Call the super parse_args method
        args = super(TreeParser, self).parse_args(*args, **kwargs)

        # Check if '--workspace_folder' has not been added manually
        if hasattr(args, "workspace_folder") and not self.added_workspace_arg_automatically:
            logger.error("Forbidden to add 'workspace_folder' argument manually. " +
                         "Can only be used with 'add_workspace_and_camera_h5_folder_arguments()'.")
            raise ValueError("Forbidden to add 'workspace_folder' argument manually. " +
                             "Can only be used with 'add_workspace_and_camera_h5_folder_arguments()'.")

        # Check if workspace folder is provided
        if hasattr(args, "workspace_folder"):

            # Workspace argument provided
            if args.workspace_folder is not None:
                args.camera_poses_path = str(Path(args.workspace_folder) / "image_poses.txt")
                args.camera_poses_format = "vulkan_text"
                args.h5_folder = str(Path(args.workspace_folder) / "output")

                logger.info(f"Using workspace folder '{args.workspace_folder}'.\nBy default using:\n" +
                            f"  - GT camera poses from '{args.camera_poses_path}'\n" +
                            f"  - H5 files from the folder '{args.h5_folder}'")

            # Workspace argument not provided -> assert that the other arugments are not None
            else:
                if args.camera_poses_path is None or args.camera_poses_format is None or args.h5_folder is None:
                    logger.error("When the --workspace_folder argument is not provided, the arguments " +
                                 "--camera_poses_path, --camera_poses_format, --h5_folder must be provided.")
                    raise ValueError("When the --workspace_folder argument is not provided, the arguments " +
                                     "--camera_poses_path, --camera_poses_format, --h5_folder must be provided.")

        return args