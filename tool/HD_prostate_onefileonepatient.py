from typing import Union, Tuple, List
import re
from PIL import Image
from deepclustering.meters import MeterInterface
import numpy as np
from torch import Tensor
from deepclustering.utils import Path
import os
import torch
from deepclustering.utils import (
    simplex,
    one_hot,
    class2one_hot,
    probs2one_hot,
)

from tool.surface_meter import SurfaceMeter

strs = '../../../../PHD_documents/dataset_slices/prostate'
patients = ['02', '16', '19', '24', '25', '26', '30', '31', '32', '48']
index1 =['Case02', 'Case16', 'Case19', 'Case24', 'Case25', 'Case26',
         'Case30', 'Case31', 'Case32', 'Case48']
C = 2

_meter_interface = MeterInterface()

for kk in range(15):
    _meter_interface.register_new_meter(
        f"hd{kk}",
        SurfaceMeter(
            C=C,
            report_axises=list(range(1, C)),
            metername="hausdorff",
        ),
        group_name="inference",
    )


def _convert2onehot(pred: Tensor, target: Tensor, C):
    # only two possibility: both onehot or both class-coded.
    assert pred.shape == target.shape
    # if they are onehot-coded:
    if simplex(pred, 1) and one_hot(target):
        return probs2one_hot(pred).long(), target.long()
    # here the pred and target are labeled long
    return (
        class2one_hot(pred, C).long(),
        class2one_hot(target, C).long(),
    )


def average_list(input_list):
    return sum(input_list) / len(input_list)


def extract_suject_num(path: Union[str, Path], indicator) -> Tuple[str, str]:
    if isinstance(path, str):
        path = Path(path)
    stem = path.stem
    subject_num = re.compile(fr"\d+").findall(stem)[2]
    return subject_num


def _glob_folder(current_folder: Path, indicator, postfix: str) -> List[str]:
    return sorted(
        set([extract_suject_num(x, indicator) for x in current_folder.rglob(f"*.{postfix}")])
    )


def ensembel3D(path, k):
    volum = []
    for img_path in path:
        indicator = f'Case_0_'
        img_list = _glob_folder(Path(img_path), indicator, 'png')
        vol_sub = []
        for slice_num in img_list:
            img_i = Image.open(f'{img_path}/{os.listdir(img_path)[0][:-6]}{slice_num}.png')
            img_slice = torch.Tensor(np.array(img_i))
            vol_sub.append(img_slice.unsqueeze(0)) #3d
        volum.append(torch.cat(vol_sub, dim=0))

    list_sub = []
    for method_i in range(len(volum)):
        if volum[method_i].unique().sum() != 0 and volum[0].unique().sum() != 0:
            onehot_pred, onehot_target = _convert2onehot(volum[method_i], volum[0], C)
            _meter_interface[f"hd{method_i}"].add(onehot_pred.transpose(1, 0).unsqueeze(0),
                                                  onehot_target.transpose(1, 0).unsqueeze(0))
            list_sub.append(_meter_interface[f"hd{method_i}"].value()[0])
        else:
            list_sub.append(0)
    return list_sub


str1 = '../../../../PHD_documents/papers_work/CaVAT_TMI/reward_types_tests/discrete_promise_runs/segmentations_runs'

HD_list = []
for i in range(len(patients)):
    gt_path = f'{strs}/gt/{patients[i]}'
    # full_path = f'{strs}/baseline_f_3/iter100/pre/{patients[i]}'
    # p_path = f'{strs}/baseline_p_3/iter100/pre/{patients[i]}'
    # entropy_path = f'{strs}/entropy_3/iter100/pre/{patients[i]}'
    # co_t_path = f'{strs}/co_training_3/iter100/pre/{patients[i]}'
    # mt_path = f'{strs}/meanteacher_3/iter100/pre/{patients[i]}'

    run1 = f'{str1}/promise_205reg_discrete_301eps_101cons_7_run1/iter099/predictions/{patients[i]}'
    run2 = f'{str1}/promise_205reg_discrete_301eps_101cons_7_run2/iter099/predictions/{patients[i]}'
    run3 = f'{str1}/promise_205reg_discrete_301eps_101cons_7_run3/iter099/predictions/{patients[i]}'

    path = [gt_path, run1, run2, run3]
    HD_score = ensembel3D(path, i)

    HD_list.append(HD_score)


s1_1 = 0
s2_1 = 0
s3_1 = 0
# s4_1 = 0
# s5_1 = 0
# s6_1 = 0
# s7_1 = 0
# s8_1 = 0
# s9_1 = 0
for method in HD_list:
    s1_1 = s1_1 + method[1]
    s2_1 = s2_1 + method[2]
    s3_1 = s3_1 + method[3]
    # s4_1 = s4_1 + method[4]
    # s5_1 = s5_1 + method[5]
    # s6_1 = s6_1 + method[6]
    # s7_1 = s7_1 + method[7]
    # s8_1 = s8_1 + method[8]
    # s9_1 = s9_1 + method[9]

s1_1 = s1_1/len(HD_list)
s2_1 = s2_1/len(HD_list)
s3_1 = s3_1/len(HD_list)
# s4_1 = s4_1/len(list_patient)
# s5_1 = s5_1/len(list_patient)
# s6_1 = s6_1/len(list_patient)
# s7_1 = s7_1/len(list_patient)
# s8_1 = s8_1/len(list_patient)
# s9_1 = s9_1/len(list_patient)


list_final = [s1_1, s2_1, s3_1]
print(list_final)




















