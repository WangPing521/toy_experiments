from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from deepclustering.utils import assert_list

__all__ = [
    "gaussian_normalize",
    "maxmin_normalization",
    "cut_edge",
    "path2Path",
    "path2str",
    "reviseResolution",
]


def gaussian_normalize(*array: np.ndarray, use_mask=True):
    for x in array:
        if use_mask:
            norm_img = _normalize_array(x, mask=x > 0)
        else:
            norm_img = x
    return norm_img


def _normalize_array(array: np.ndarray, mask=None):
    # Normalization
    if mask is not None:
        array_norm = (array - array[mask].mean()) / array[mask].std()
        return array_norm
    return (array - array.mean()) / array.std()


def _maxmin_normalization(array: np.ndarray, pertile=0.95):
    minthreshold = np.percentile(array.ravel(), (1 - pertile) * 100)
    maxthreshold = np.percentile(array.ravel(), pertile * 100)
    new_array = array.copy().astype(np.float)
    new_array[new_array > maxthreshold] = maxthreshold
    new_array[new_array < minthreshold] = minthreshold
    norm_arry = (new_array - minthreshold) / (maxthreshold - minthreshold)
    assert (norm_arry >= 0).all() & (norm_arry <= 1).all()
    return norm_arry


def maxmin_normalization(*args, pertile=0.95):
    for x in args:
        img = _maxmin_normalization(x, pertile=pertile)
    return img


def cut_edge(
    *images: np.ndarray, label: np.ndarray, margin=((16, 16), (16, 16), (16, 16))
):
    t1 = images[0]
    padded_t1_nonzeros = (t1 - t1.min()).nonzero()
    h_min, h_max = padded_t1_nonzeros[0].min(), padded_t1_nonzeros[0].max()
    w_min, w_max = padded_t1_nonzeros[1].min(), padded_t1_nonzeros[1].max()
    d_min, d_max = padded_t1_nonzeros[2].min(), padded_t1_nonzeros[2].max()
    cropped_images = [x[h_min:h_max, w_min:w_max, d_min:d_max] for x in images]
    cropped_label = label[h_min:h_max, w_min:w_max, d_min:d_max]
    return (
        *[np.pad(x, margin, constant_values=x[10][10][10]) for x in cropped_images],
        np.pad(cropped_label, margin, constant_values=0),
    )


def path2Path(path) -> Path:
    assert isinstance(path, (str, Path)), type(path)
    return path if isinstance(path, Path) else Path(path)


def path2str(path):
    return str(path)


def reviseResolution(*images, label, resolution=(0.958, 0.958, 1)):
    assert assert_list(lambda x: isinstance(x, np.ndarray), images)
    resized_images = [zoom(input=x, zoom=resolution, order=3) for x in images]
    resized_labels = zoom(input=label.astype(np.float), zoom=resolution, order=0)
    assert np.allclose(np.unique(resized_labels), np.unique(label))
    return (*resized_images, resized_labels)