import re
from pathlib import Path
from pprint import pprint
from typing import Union, List, Tuple

import h5py
import numpy as np
from medpy.io import load


def extract_suject_stem(path: Union[str, Path]) -> Tuple[str, str]:
    if isinstance(path, str):
        path = Path(path)
    stem = path.stem
    subject_num = re.compile(r"subject-\d+").findall(stem)[0].split("-")[1]
    image_type = re.compile(r"\d+-.*").findall(stem)[0].split("-")[1]
    return subject_num, image_type


def convert_label(label_img):
    label_processed = np.zeros(label_img.shape[0:]).astype(np.uint8)
    for i in range(label_img.shape[2]):
        label_slice = label_img[:, :, i]
        label_slice[label_slice == 10] = 1
        label_slice[label_slice == 150] = 2
        label_slice[label_slice == 250] = 3
        label_processed[:, :, i] = label_slice
    return label_processed


class BuidDataset:
    def __init__(self, data_root: Union[str, Path], dataset_postfix="hdr") -> None:
        super().__init__()
        self._data_root: Path = Path(data_root) if isinstance(
            data_root, str
        ) else data_root
        self._post_fix = dataset_postfix
        self._content_list = self._glob_folder(self._data_root, self._post_fix)
        assert len(self._content_list) > 0, self._content_list
        pprint(self._content_list[:5])

    @staticmethod
    def _glob_folder(current_folder: Path, postfix: str) -> List[str]:
        return sorted(
            set(
                [
                    "subject-%s" % extract_suject_stem(x)[0]
                    for x in current_folder.rglob(f"*.{postfix}")
                ]
            )
        )

    def _load_numpy_data(
        self, subject_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        f_T1 = self._data_root / (subject_name + "-T1.hdr")
        assert f_T1.exists() and f_T1.is_file(), f_T1
        img_T1, header_T1 = load(str(f_T1))
        f_T2 = self._data_root / (subject_name + "-T2.hdr")
        img_T2, header_T2 = load(str(f_T2))
        try:
            f_l = self._data_root / (subject_name + "-label.hdr")
            labels, header_label = load(str(f_l))
        except:
            labels = np.zeros_like(img_T1)
        inputs_T1 = img_T1.astype(np.float32)
        inputs_T2 = img_T2.astype(np.float32)
        inputs_T1_norm = self._normalize_array(inputs_T1, mask=inputs_T1 > 0)
        inputs_T2_norm = self._normalize_array(inputs_T2, mask=inputs_T1 > 0)
        labels = convert_label(labels).astype(np.uint8)

        return inputs_T1_norm, inputs_T2_norm, labels

    @staticmethod
    def _normalize_array(array: np.ndarray, mask=None):
        # Normalization
        if mask is not None:
            array_norm = (array - array[mask].mean()) / array[mask].std()
            return array_norm
        return (array - array.mean()) / array.std()

    @staticmethod
    def _cut_edge(
        t1: np.ndarray, t2: np.ndarray, label: np.ndarray, margin=32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert t1.shape == t2.shape == label.shape
        padded_t1_nonzeros = (t1 - t1.min()).nonzero()
        h_min, h_max = padded_t1_nonzeros[0].min(), padded_t1_nonzeros[0].max()
        w_min, w_max = padded_t1_nonzeros[1].min(), padded_t1_nonzeros[1].max()
        d_min, d_max = padded_t1_nonzeros[2].min(), padded_t1_nonzeros[2].max()
        cropped_t1 = t1[h_min:h_max, w_min:w_max, d_min:d_max]
        cropped_t2 = t2[h_min:h_max, w_min:w_max, d_min:d_max]
        cropped_label = label[h_min:h_max, w_min:w_max, d_min:d_max]
        return (
            np.pad(cropped_t1, margin, constant_values=t1.min()),
            np.pad(cropped_t2, margin, constant_values=t2.min()),
            np.pad(cropped_label, margin, constant_values=0),
        )

    def write_data2h5py(self, save_dir):
        writer = H5pyWriter(save_dir)
        for subject in self._content_list:
            t1, t2, label = self._load_numpy_data(subject)
            t1, t2, label = self._cut_edge(t1, t2, label)
            writer.write(t1, t2, label, subject)


class H5pyWriter:
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self._save_dir: Path = save_dir if isinstance(save_dir, Path) else Path(
            save_dir
        )
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def write(self, t1, t2, label, subject_name: str):
        inputs_tmp_T1 = t1[:, :, :, None]
        inputs_tmp_T2 = t2[:, :, :, None]
        labels_tmp = label[:, :, :, None]

        inputs = np.concatenate((inputs_tmp_T1, inputs_tmp_T2), axis=3)
        inputs_caffe = inputs[None, :, :, :, :]
        labels_caffe = labels_tmp[None, :, :, :, :]

        inputs_caffe = inputs_caffe.transpose(0, 4, 3, 1, 2)
        labels_caffe = labels_caffe.transpose(0, 4, 3, 1, 2)
        print(inputs_caffe.shape, labels_caffe.shape)
        with h5py.File(str(self._save_dir / subject_name) + ".h5", "w") as f:
            f["data"] = inputs_caffe  # for caffe num channel x d x h x w
            f["label"] = labels_caffe


if __name__ == "__main__":
    DATA_PATH = '../dataset/iSeg2017'
    dataset = BuidDataset(Path(DATA_PATH, "iSeg-2017-Training"))
    dataset.write_data2h5py(Path(DATA_PATH, "iSeg-2017-Training-h5py"))
    dataset = BuidDataset(Path(DATA_PATH, "iSeg-2017-Testing"))
    dataset.write_data2h5py(Path(DATA_PATH, "iSeg-2017-Testing-h5py"))