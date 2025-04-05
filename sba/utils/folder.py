# Python
import os, os.path, shutil
from typing import Union
from pathlib import Path, PosixPath, WindowsPath

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

def create_folders(*args: Union[str, Path]) -> None:
    for folder in args:
        folder = Path(folder)
        assert type(folder) == PosixPath or type(folder) == WindowsPath
        create_folder(folder)

def create_folder(folder: Union[str, Path]) -> None:
    # Turn into Path
    folder = Path(folder)

    # If the folder does not exist, create it
    if not folder.exists():
        folder.mkdir()
    # If it already exists, empty its contents
    else:
        for filename in os.listdir(str(folder)):
            file_path = os.path.join(str(folder), filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Error deleting {file_path}: {e}")