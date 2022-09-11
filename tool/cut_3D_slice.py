import re
from pathlib import Path
from pprint import pprint
from typing import Union, Tuple, List
import numpy as np
from medpy.io import load


# Reference https://github.com/zhengyang-wang/Unet_3D/tree/master/preprocessing
from tool._utils import cut_edge, maxmin_normalization, gaussian_normalize, path2Path, reviseResolution
from tool._writer import H5pyWriter, SliceWriter


def extract_suject_num(path: Union[str, Path], indicator) -> Tuple[str, str]:
    if isinstance(path, str):
        path = Path(path)
    stem = path.stem
    subject_num = re.compile(fr"{indicator}_\d+").findall(stem)[0].split("_")[2]
    return subject_num


def convert_label(label_img):
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:, :, i]
        label_slice[label_slice == 38] = 1
        label_slice[label_slice == 52] = 2
        label_slice[label_slice == 82] = 3
        label_slice[label_slice == 88] = 4
        label_slice[label_slice == 164] = 5
        label_slice[label_slice == 205] = 6
        label_slice[label_slice == 244] = 7
        label_processed[:, :, i] = label_slice
    return label_processed


class BuidDataset:
    def __init__(self, data_root: Union[str, Path], indicator, dataset_postfix="nii.gz") -> None:
        super().__init__()
        self._data_root: Path = path2Path(Path(data_root, indicator))
        assert self._data_root.exists(), self._data_root
        self.indicator = indicator
        self._post_fix = dataset_postfix
        self._content_list = self._glob_folder(self._data_root, indicator, self._post_fix)
        assert len(self._content_list) > 0, self._content_list
        pprint(self._content_list)

    @staticmethod
    def _glob_folder(current_folder: Path, indicator, postfix: str) -> List[str]:
        return sorted(
            set([extract_suject_num(x, indicator) for x in current_folder.rglob(f"*.{postfix}")])
        )

    def _load_numpy_data(
        self, subject_name: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img = self._data_root / f'{self.indicator}_{subject_name}_image.nii.gz'
        assert img.exists() and img.is_file(), img
        img, _ = load(str(img))
        img_label = self._data_root / f'{self.indicator}_{subject_name}_label.nii.gz'
        if img_label.exists():
            labels, _ = load(str(img_label))

        img = img.astype(np.float32)
        img_label = convert_label(np.float32(labels.astype(np.uint8)))

        assert img.shape == img_label.shape

        return img, img_label

    def write_data2h5py(self, save_dir):
        writer = H5pyWriter(save_dir)
        for subject in self._content_list:
            mr_img, label = self._load_numpy_data(subject)
            mr_img, label = cut_edge(
                mr_img, label=label, margin=((0, 0), (0, 0), (0, 0))
            )
            mr_img = gaussian_normalize(mr_img)

            writer.write(mr_img, label=label, subject_name=subject)

    def slice_and_save(self, save_dir):
        # slice dataset to 2d and save in a structured way.
        writer = SliceWriter(save_dir)
        for subject in self._content_list:
            mr_img, label = self._load_numpy_data(subject)
            mr_img, label = cut_edge(
                mr_img, label=label, margin=((32, 32), (32, 32), (0, 0))
            )
            mr_img = maxmin_normalization(mr_img, pertile=0.999)
            writer.write(
                mr_img=mr_img, label=label, subject_name=subject
            )


if __name__ == "__main__":
    DATA_PATH='../dataset/WMH'
    indicator = 'ct_train'
    # dataset = BuidDataset(Path(DATA_PATH), indicator)
    # dataset.write_data2h5py(Path(DATA_PATH, "mr_train_h5py"))
    # dataset.slice_and_save(Path(DATA_PATH, "mr_train_2D"))

    dataset = BuidDataset(Path(DATA_PATH), indicator)
    # dataset.write_data2h5py(Path(DATA_PATH, "ct_train_h5py"))
    dataset.slice_and_save(Path(DATA_PATH, "ct_train_2D"))
