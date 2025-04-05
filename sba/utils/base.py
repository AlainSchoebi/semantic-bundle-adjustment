# Typing
from __future__ import annotations
from typing import Callable, Any, Tuple

# Numpy
import numpy as np
from numpy.typing import NDArray

def dehomogenized(vectors: NDArray) -> NDArray:
    """
    Dehomogenise vectors stored in matrix (..., d + 1), scaling down by the last element of each vector and returning a
    d-dimensional homogeneous vectors in matrix of size (..., d).
    """
    return vectors[..., :-1] / vectors[..., -1:]

def homogenized(vectors: NDArray, fill_value: Any = 1) -> NDArray:
    """
    Homogenise d-dimensional vectors stored in matrix (..., d), returning homogeneous vectors in matrix of size
    (..., d + 1).
    """
    return np.concatenate((vectors, np.full(vectors.shape[:-1], fill_value, dtype=vectors.dtype)[..., np.newaxis]), axis=-1, dtype=vectors.dtype)

def normalized(x: NDArray, axis = -1) -> NDArray:
    """
    Normalize x along the provided axis. By default performs the normalization along the last axis.
    """
    if np.any(np.linalg.norm(x, axis=axis, keepdims=True) == 0):
        raise ValueError("Error in vector normalization as norm is zero.")
    return x / np.linalg.norm(x, axis=axis, keepdims=True)

class XYWH:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def corner_coordinates(self) -> NDArray[np.int_]:
        return np.array([[self.x, self.y], [self.x, self.y + self.h - 1], [self.x + self.w - 1, self.y + self.h - 1],
                         [self.x + self.w - 1, self.y]], dtype=int)


def ransac(model_fct: Callable[[NDArray], NDArray], error_fct: Callable[[NDArray, Any], NDArray],
           data: NDArray, n: int, threshold: float, outlier_ratio: int = None, n_iterations: int = None) \
            -> Tuple[Any, NDArray]:

    """
    Uses the RANSAC method to find the best model and reject the outliers data points.

    Inputs
    - model_fct:    function that finds a model given some data, i.e model_fct(x: NDArray) -> Any
    - error_fct:    function that computes the error for every datapoint given a model,
                    i.e. error_fct(x: NDArray, model: Any) -> NDArray
    - data:         (N, ...) the data points of this problem stored as rows
    - n:            the minimum number of data points to estimate the model
    - threshold:    trehsold value that determines if the model fits a data point well or not
    - outlier_ratio:the estimated outlier ratio. If None, then uses n_iterations
    - n_iterations: the number of iterations that will be performed. If None, then uses a adaptive RANSAC method
    Returns
    - best_model:   the model that fits the data the best
    - inlier_mask: (N, ) boolean array mask for the inliers
    """

    assert len(data) >= n and "Fewer datapoints than the size of the subsets needed to estimate the model."

    if outlier_ratio != None:
        prob_success = 0.99
        n_iterations = int(np.ceil(np.log(1 - prob_success) / np.log(1 - (1 - outlier_ratio)**n)))
    elif n_iterations == None:
        raise NotImplementedError("Adapative RANSAC is not implemented.")

    max_n_inliers = 0
    best_model, best_inlier_mask = None, None
    for _ in range(n_iterations):

        # take a subset
        subset_idxs = np.random.choice(len(data), size = n, replace = False)
        data_sub = data[subset_idxs]

        # find the model, get the error and set the inlier mask
        model = model_fct(data_sub)    # Any
        error = error_fct(data, model) # (N, )

        inlier_mask = error < threshold

        # count number of inliers and update the best model if necessary
        n_inliers = np.sum(inlier_mask)
        if n_inliers > max_n_inliers:
            max_n_inliers = n_inliers
            best_model, best_inlier_mask = model, inlier_mask

    return best_model, best_inlier_mask
