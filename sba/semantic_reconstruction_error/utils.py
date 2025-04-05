# Numpy
import numpy as np
from numpy.typing import NDArray
import numpy.ma as ma
from numpy.ma import MaskedArray

# Python
from typing import List, Tuple, Type, Literal

# Src
from sba.constants import semantic_reassign, SemanticError

# Utils
from sba.utils.cameras import PinholeCamera

# Logging
from sba.utils.loggers import get_logger
logger = get_logger(__name__)

def compare_semantics(semantics_1: NDArray, semantics_2: NDArray, camera_1: PinholeCamera, camera_2: PinholeCamera) -> SemanticError:

    """
    Compare two semantics vector, where semantics_1 is considered as the base vector and semantics_2 the vector to compare to.
    """

    # Reassign the semantic classes if needed
    semantics_1, semantics_2 = semantic_reassign(semantics_1), semantic_reassign(semantics_2)

    # Confusion matrix
    classes = np.unique(np.concatenate((np.unique(semantics_1), np.unique(semantics_2)))) # C unique classes
    mask = semantics_1[:, None] == classes # (N, C) N: number of pixels, C: number of classes
    mm = ma.array(semantics_2[:, None][:, [0]*len(classes)], mask = ~mask) # (N, C)
    corresponding_classes = mm[..., None] == classes # (N, C, C)
    confusion_matrix = ma.sum(corresponding_classes, axis = 0).filled(0)

    # Compute error
    semantic_error = semantics_2 != semantics_1 # (?''',)

    # Log
    logger.debug(f"**When reprojecting from {camera_1.name} onto {camera_2.name}: " +
                    f"{np.sum(semantic_error)/semantic_error.size*100:.1f}% of the pixels " +
                    f"(out of {semantic_error.size:.1e}) have wrong semantic classes. **")
    logger.debug(f"The confusion matrix, for the classes {classes}, is given by:\n{confusion_matrix}\n")

    error: SemanticError = ((camera_1.name, camera_2.name), (np.sum(semantic_error), semantic_error.size))
    return error