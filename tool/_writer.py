import warnings
from itertools import repeat
from pathlib import Path
from typing import *

import h5py
import numpy as np
from skimage.io import imsave

__all__ = ["H5pyWriter", "SliceWriter"]


class H5pyWriter:
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self._save_dir: Path = save_dir if isinstance(save_dir, Path) else Path(
            save_dir
        )
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def write(self, *images, label, subject_name: str):
        inputs_tmp = []
        for img in images:
            inputs_tmp.append(img[:, :, :, None])
        labels_tmp = label[:, :, :, None]

        inputs = np.concatenate(inputs_tmp, axis=3)
        inputs_caffe = inputs[None, :, :, :, :]
        labels_caffe = labels_tmp[None, :, :, :, :]

        inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
        labels_caffe = labels_caffe.transpose(0, 4, 3, 1, 2)
        print(inputs_caffe.shape, labels_caffe.shape)
        with h5py.File(str(self._save_dir / subject_name) + ".h5", "w") as f:
            f["data"] = inputs_caffe  # for caffe num channel x d x h x w
            f["label"] = labels_caffe


class SliceWriter:
    def __init__(self, save_dir) -> None:
        self._save_dir = Path(save_dir)

    def write(
        self,
        mr_img: np.ndarray = None,
        label: np.ndarray = None,
        subject_name: str = None,
    ):
        save_path = self._save_dir
        label_path = save_path / "gt"
        label_path.mkdir(exist_ok=True, parents=True)
        if mr_img is not None:
            img_path = save_path / "img"
            img_path.mkdir(exist_ok=True, parents=True)
        for num, (_mr_img, _lab) in enumerate(
            zip(
                mr_img.transpose(2, 0, 1) if mr_img is not None else repeat(None),
                label.transpose(2, 0, 1),
            )
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if _mr_img is not None:
                    imsave(
                        str(img_path / f"{subject_name}_{num:03d}.png"),
                        (_mr_img * 255).astype(np.uint8),
                    )
                imsave(
                    str(label_path / f"{subject_name}_{num:03d}.png"),
                    _lab.astype(np.uint8),
                )