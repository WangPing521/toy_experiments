import os
import re
from pathlib import Path
from pprint import pprint
from typing import Union, List, Tuple
import numpy as np
from medpy.io import load
import h5py
from scipy.ndimage import zoom
import torch
from skimage.io import imsave

from tool._utils import _maxmin_normalization


def path2Path(path) -> Path:
    assert isinstance(path, (str, Path)), type(path)
    return path if isinstance(path, Path) else Path(path)


def gaussian_normalize(array: np.ndarray, use_mask=True):
    return _normalize_array(array, mask=array > 0 if use_mask else None)


def maxmin_normalization(*args, pertile=0.95):
    for x in args:
        img = _maxmin_normalization(x, pertile=pertile)
    return img


def _normalize_array(array: np.ndarray, mask=None):
    # Normalization
    if mask is not None:
        array_norm = (array - array[mask].mean()) / array[mask].std()
        return array_norm
    return (array - array.mean()) / array.std()


def extract_suject_num(path: Union[str, Path]) -> str:
    path = str(path)
    # Spleen
    subject_num = re.compile(r"\d+").findall(path)[1].split("/")[0]

    return subject_num


def reviseResolution(images, label, resolution=(0.958, 0.958, 1)):
    resized_images = zoom(input=images, zoom=resolution, order=3)
    resized_labels = zoom(input=label.astype(np.float), zoom=resolution, order=0)
    assert np.allclose(np.unique(resized_labels), np.unique(label))
    return resized_images, resized_labels


def cut_niiedge(
    img: np.ndarray, label: np.ndarray, margin=8
) -> Tuple[np.ndarray, np.ndarray]:
    assert img.shape == label.shape
    center_region = label.nonzero()
    d_min, d_max = center_region[0].min(), center_region[0].max()
    # h_min, h_max = center_region[1].min(), center_region[1].max()
    # w_min, w_max = center_region[2].min(), center_region[2].max()
    margin_ds = min(d_min-0, margin)
    margin_de = min(img.shape[0] - d_max, margin)

    cropped_img = img[d_min - margin_ds:d_max + margin_de, :, :]
    cropped_label = label[d_min - margin_ds:d_max + margin_de, :, :]
    return cropped_img, cropped_label


def convert_label(label_img):
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:, :, i]
        label_slice[label_slice == 0] = 0
        label_slice[label_slice == 1] = 1
        label_slice[label_slice == 2] = 2
        label_processed[:, :, i] = label_slice
    return label_processed


class H5pyWriter:
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self._save_dir: Path = save_dir if isinstance(save_dir, Path) else Path(
            save_dir
        )
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def write(self, img, label, subject_name: str):

        inputs_tmp = img[:, :, :, None]
        labels_tmp = label[:, :, :, None]

        inputs_caffe = inputs_tmp[None, :, :, :, :]
        labels_caffe = labels_tmp[None, :, :, :, :]

        inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
        labels_caffe = labels_caffe.transpose(0, 4, 3, 1, 2)
        print(inputs_caffe.shape, labels_caffe.shape)
        with h5py.File(str(self._save_dir / subject_name) + ".h5", "w") as f:
            f["data"] = inputs_caffe  # for caffe num channel x d x h x w
            f["label"] = labels_caffe


class slice2dWriter:
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self._save_dir: Path = save_dir if isinstance(save_dir, Path) else Path(
            save_dir
        )
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def write(self, img, label, subject_name: str):
        assert img.shape == label.shape
        num = img.shape[0]
        for i in range(num):
            save_img_path = Path(self._save_dir, f"img/hippocampus_{subject_name}_{i}").with_suffix(".png")
            save_gt_path = Path(self._save_dir, f"gt/hippocampus_{subject_name}_{i}").with_suffix(".png")
            save_img_path.parent.mkdir(parents=True, exist_ok=True)
            save_gt_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_img_path), (img[i] * 255).astype(np.uint8))
            imsave(str(save_gt_path), label[i].astype(np.uint8))


class BuidDataset:
    def __init__(self, data_root: Union[str, Path], dataset_postfix="png") -> None:
        super().__init__()
        self._data_root: Path = path2Path(data_root)
        assert self._data_root.exists(), self._data_root
        self._post_fix = dataset_postfix
        self._content_list = self._glob_folder(self._data_root, self._post_fix)
        assert len(self._content_list) > 0, self._content_list
        pprint(self._content_list)

    @staticmethod
    def _glob_folder(current_folder: Path, postfix: str) -> List[str]:
        return sorted(
            set([extract_suject_num(x) for x in current_folder.rglob(f"*.{postfix}")])
        )

    def _load_numpy_data(
        self, subject_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # scan_img = self._data_root / 'img' / f'prostate_{subject_name}.nii.gz'
        scan_img = self._data_root / 'img' / f'hippocampus_{subject_name}.nii.gz'

        assert scan_img.exists() and scan_img.is_file(), scan_img
        img, _ = load(str(scan_img))
        img = img.astype(np.float32).transpose(2, 1, 0)
        scan_label = self._data_root / 'gt' / f'hippocampus_{subject_name}.nii.gz'
        if scan_label.exists():
            labels, _ = load(str(scan_label))
            labels = convert_label(np.float32(labels.astype(np.uint8))).transpose(2, 1, 0)
        assert img.shape == labels.shape

        return img, labels

    def write_data2h5py(self, save_dir):
        writer = H5pyWriter(save_dir)
        for subject in self._content_list:
            img, label = self._load_numpy_data(subject)
            # indexes = torch.where(torch.Tensor(label) == 1)
            # img, label = cut_edge(img, label=label, margin=indexes)
            img, label = reviseResolution(
                img, label=label, resolution=(192 / img.shape[0], 192 / img.shape[1], 192 / img.shape[2])
            )

            img = gaussian_normalize(img)

            writer.write(img, label=label, subject_name=subject)

    def slice_write_2d(self, save_dir):
        writer = slice2dWriter(save_dir)
        for subject in self._content_list:
            img, label = self._load_numpy_data(subject)
            img, label = cut_niiedge(img, label)
            # img, label = cut_edge(img, label=label, margin=[8,8,8])
            img, label = reviseResolution(
                img, label=label, resolution=(1, 96 / img.shape[1], 96 / img.shape[2])
            )

            img = maxmin_normalization(img, pertile=0.999)

            # writer.write(img, label=label, subject_name=subject)


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), "dataset")
    dataset = BuidDataset(
        Path(DATA_PATH, "Task04_Hippocampus/train"), dataset_postfix="nii.gz"
    )
    dataset.slice_write_2d(Path(DATA_PATH, "Hipocampus/train"))