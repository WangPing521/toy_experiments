import os
import re
from pathlib import Path
from typing import Union, List, Tuple

import numpy as np
from medpy.io import load
from scipy.ndimage import zoom
import torch

# for MRbrain dataset
# Algorithms that only segment gray matter, white matter and cerebrospinal fluid should merge labels 1 and 2,
# 3 and 4, and 5 and 6, and label the output as either 0 (background), 1 (gray matter), 2 (white matter) and 3 (CSF).
# The cerebellum and brain stem (label 7 and 8) will in that case be excluded from the evaluation.
from tool._writer import H5pyWriter


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
        # GM
        label_slice[label_slice == 1] = 1
        label_slice[label_slice == 2] = 1
        # WM
        label_slice[label_slice == 3] = 2
        label_slice[label_slice == 4] = 2
        # CSF
        label_slice[label_slice == 5] = 3
        label_slice[label_slice == 6] = 3

        label_processed[:, :, i] = label_slice
    return label_processed


class BuidDataset:
    def __init__(self, data_root: Union[str, Path], img='pre', dataset_postfix="nii") -> None:
        super().__init__()
        self._data_root: Path = Path(data_root) if isinstance(
            data_root, str
        ) else data_root
        self.imgdir = img
        self._post_fix = dataset_postfix
        subject = os.listdir(DATA_PATH)
        self.subject_list = subject

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

    def class2one_hot(self, seg, C, class_dim: int = 1):
        seg = torch.Tensor(seg).type(torch.float)
        if len(seg.shape) == 3:
            seg = seg.unsqueeze(dim=0)
        res: torch.Tensor = torch.stack([seg == c for c in range(C)], dim=class_dim).type(torch.float)
        return res

    def reviseResolution(self, T1, T1_IR, FLAIR, label):
        T1_1 = zoom(T1, (0.958, 0.958, 3))
        T1_IR_1 = zoom(T1_IR, (0.958, 0.958, 3))
        FLAIR_1 = zoom(FLAIR, (0.958, 0.958, 3))
        labels_class = convert_label(label).astype(np.uint8)
        labels_onehot = self.class2one_hot(labels_class, 6) # 0,1,2,3,7,8
        labels_onehot = labels_onehot.squeeze(0)
        labels_origin = np.concatenate(
            (torch.Tensor(labels_onehot[0]).unsqueeze(0), torch.Tensor(labels_onehot[1]).unsqueeze(0), torch.Tensor(labels_onehot[2]).unsqueeze(0),
             torch.Tensor(labels_onehot[3]).unsqueeze(0)), axis=0)
        label_1_origin = torch.Tensor(labels_origin.transpose(0,3,1,2)).unsqueeze(0).max(1)[1].squeeze(0)
        labels_onehot1 = torch.Tensor(zoom(labels_onehot[0], (0.958, 0.958, 3), order=0))
        labels_onehot2 = torch.Tensor(zoom(labels_onehot[1], (0.958, 0.958, 3), order=0))
        labels_onehot3 = torch.Tensor(zoom(labels_onehot[2], (0.958, 0.958, 3), order=0))
        labels_onehot4 = torch.Tensor(zoom(labels_onehot[3], (0.958, 0.958, 3), order=0))
        labels = np.concatenate(
            (labels_onehot1.unsqueeze(0), labels_onehot2.unsqueeze(0), labels_onehot3.unsqueeze(0), labels_onehot4.unsqueeze(0)), axis=0
        )
        label_1 = torch.Tensor(labels).unsqueeze(0).max(1)[1].squeeze(0)
        return T1_1, T1_IR_1, FLAIR_1, label_1

    def _load_numpy_data(
        self, subject_path: str, indicator: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data_path = os.path.join(subject_path, indicator)
        label_path = subject_path

        f_T1 = Path(os.path.join(data_path, "reg_T1.nii.gz"))
        assert f_T1.exists() and f_T1.is_file(), f_T1
        img_T1, header_T1 = load(str(f_T1))

        f_FlAIR = Path(os.path.join(data_path, "FLAIR.nii.gz"))
        assert f_FlAIR.exists() and f_FlAIR.is_file(), f_FlAIR
        img_FlAIR, header_FlAIR = load(str(f_FlAIR))

        f_IR = Path(os.path.join(data_path, "reg_IR.nii.gz"))
        assert f_IR.exists() and f_IR.is_file(), f_IR
        img_IR, header_IR = load(str(f_IR))

        try:
            f_l = Path(os.path.join(label_path, "segm.nii.gz"))
            labels, header_label = load(str(f_l))
        except:
            labels = np.zeros_like(img_T1)

        inputs_T1 = img_T1.astype(np.float32)
        inputs_FlAIR = img_FlAIR.astype(np.float32)
        inputs_IR = img_IR.astype(np.float32)

        inputs_T1_1, inputs_FlAIR_1, inputs_IR_1, labels_1 = self.reviseResolution(inputs_T1, inputs_FlAIR, inputs_IR, labels)

        inputs_T1_norm = self._normalize_array(inputs_T1_1, mask=inputs_T1_1 > 0)
        inputs_FlAIR_norm = self._normalize_array(inputs_FlAIR_1, mask=inputs_FlAIR_1 > 0)
        inputs_IR_norm = self._normalize_array(inputs_IR_1, mask=inputs_IR_1 > 0)


        # assert (
        #     inputs_T1_norm.shape
        #     == inputs_T2_norm.shape
        #     == inputs_T1_IR_norm.shape
        #     == labels.shape
        # )
        return inputs_T1_norm, inputs_FlAIR_norm, inputs_IR_norm, labels_1

    @staticmethod
    def _normalize_array(array: np.ndarray, mask=None):
        # Normalization
        if mask is not None:
            array_norm = (array - array[mask].mean()) / array[mask].std()
            return array_norm
        return (array - array.mean()) / array.std()

    @staticmethod
    def _cut_edge(
        t1: np.ndarray, t1_ir: np.ndarray, t2: np.ndarray, label: np.ndarray, margin=32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert t1.shape == t2.shape == label.shape == t1_ir.shape
        padded_t1_nonzeros = (t1 - t1.min()).nonzero()
        h_min, h_max = padded_t1_nonzeros[0].min(), padded_t1_nonzeros[0].max()
        w_min, w_max = padded_t1_nonzeros[1].min(), padded_t1_nonzeros[1].max()
        d_min, d_max = padded_t1_nonzeros[2].min(), padded_t1_nonzeros[2].max()
        cropped_t1 = t1[h_min:h_max, w_min:w_max, d_min:d_max]
        cropped_t1_ir = t1_ir[h_min:h_max, w_min:w_max, d_min:d_max]
        cropped_t2 = t2[h_min:h_max, w_min:w_max, d_min:d_max]
        cropped_label = label[h_min:h_max, w_min:w_max, d_min:d_max]
        return (
            np.pad(cropped_t1, margin, constant_values=t1.min()),
            np.pad(cropped_t1_ir, margin, constant_values=t1_ir.min()),
            np.pad(cropped_t2, margin, constant_values=t2.min()),
            np.pad(cropped_label, margin, constant_values=0),
        )

    def write_data2h5py(self, save_dir):
        writer = H5pyWriter(save_dir)
        for subject in self.subject_list:
            subject_path = os.path.join(self._data_root, subject)
            T1, T1_IR, FLAIR, label = self._load_numpy_data(subject_path, self.imgdir)
            t1, t1_ir, t2, label = self._cut_edge(T1, T1_IR, FLAIR, label)
            writer.write(t1, t1_ir, t2, label, subject)


if __name__ == "__main__":
    DATA_PATH = '../dataset/MRbrain/training'
    dataset = BuidDataset(Path(DATA_PATH), img='pre', dataset_postfix="nii")
    dataset.write_data2h5py(Path(DATA_PATH, "pre-h5py"))