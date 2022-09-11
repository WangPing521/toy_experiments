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


strs = '../../../../PHD_documents/papers_work/first_paper_TMI_MIA/TMI_200802-201116/major_revision/new_experiments/uncertaintyMT/acdc'
patients = ['006', '013', '018', '019', '028', '033', '034', '037', '039', '040', '046',
            '050', '052', '054', '062', '063', '065', '066', '069', '075', '078', '080',
            '093', '098']
index1 = ['01', '06', '09', '10', '11', '12', '13', '14']
C = 2

_meter_interface = MeterInterface()

for kk in range(12):
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


def extract_suject_num(path: Union[str, Path], indicator) -> Tuple[str, str]:
    if isinstance(path, str):
        path = Path(path)
    stem = path.stem
    subject_num = re.compile(fr"\d+").findall(stem)[3]
    return subject_num


def _glob_folder(current_folder: Path, indicator, postfix: str) -> List[str]:
    return sorted(
        set([extract_suject_num(x, indicator) for x in current_folder.rglob(f"*.{postfix}")])
    )


def average_list(input_list):
    return sum(input_list) / len(input_list)


def ensembel3D(path, k, pxiel_wise):
    volum = []
    for img_path in path:
        vol_sub = []
        num_group = []
        for file_png in os.listdir(img_path):
            num_group.append(file_png[11:13])
        group = list(set(num_group))
        for num_group in group:
            vol = []
            indicator = f'patient{patients[k]}_{num_group}_0'
            img_list = _glob_folder(Path(img_path), indicator, 'png')
            for img_num in img_list:
                img_i = Image.open(f'{img_path}/{indicator}_{img_num}.png')
                img_slice = torch.Tensor(np.array(img_i))
                if pxiel_wise:
                    img_slice = torch.where(img_slice > 0, torch.tensor([1]), torch.tensor([0]))
                vol.append(img_slice.unsqueeze(0))
            vol_sub.append(torch.cat(vol, dim=0)) # elements in vol_sub are 3D volume
        if img_path.find('/gt/') != -1:
            fg_bgs = []
            for gt_i in range(len(vol_sub)):
                # LV
                # fg_bg = torch.where(vol_sub[gt_i] == 3, torch.Tensor([1]), torch.Tensor([0]))
                # rv
                fg_bg = torch.where(vol_sub[gt_i] == 1, torch.Tensor([1]), torch.Tensor([0]))
                # Myo
                # fg_bg = torch.where(vol_sub[gt_i] == 2, torch.Tensor([1]), torch.Tensor([0]))
                fg_bgs.append(fg_bg)
            assert fg_bgs.__len__() == vol_sub.__len__()
            vol_sub = fg_bgs
        volum.append(vol_sub) # volum: all 3D volumes for all methods
    list_methods = []
    for method_i in range(len(volum)-1):
        list_sub = []
        for group_i in range(len(volum[method_i])):
            gt = volum[0][group_i]
            if volum[method_i+1][group_i].unique().sum() != 0 and gt.unique().sum() != 0:
                onehot_pred, onehot_target = _convert2onehot(volum[method_i+1][group_i], gt, C)
                _meter_interface[f"hd{method_i}"].add(onehot_pred.transpose(1, 0).unsqueeze(0), onehot_target.transpose(1, 0).unsqueeze(0))
                list_sub.append(_meter_interface[f"hd{method_i}"])
        list_methods.append(list_sub)
    return list_methods


# run1
str1 = '../../../../PHD_documents/papers_work/MICCAI2021/MICCAI_accept_results/segmentations/RV_segmentations_0.05'

HD_list = []
for i in range(len(patients)):
    gt_path = f'{strs}/gt/{patients[i]}'
    # f_path1 = f'{str1}/baseline_f_run1/iter100/predictions/{patients[i]}'
    # f_path2 = f'{str1}/baseline_f_run2/iter100/predictions/{patients[i]}'
    # f_path3 = f'{str1}/baseline_f_run3/iter100/predictions/{patients[i]}'
    # #
    # p_path1 = f'{str1}/baseline_0.03p_run1/iter100/predictions/{patients[i]}'
    # p_path2 = f'{str1}/baseline_0.03p_run2/iter100/predictions/{patients[i]}'
    # p_path3 = f'{str1}/baseline_0.03p_run3/iter100/predictions/{patients[i]}'
    # #
    # Ent_path1 = f'{str1}/EntMin_301reg_0.03run1/iter100/predictions/{patients[i]}'
    # Ent_path2 = f'{str1}/EntMin_301reg_0.03run2/iter100/predictions/{patients[i]}'
    # Ent_path3 = f'{str1}/EntMin_301reg_0.03run3/iter100/predictions/{patients[i]}'

    # vat_path1 = f'{str1}/vat_305reg1eps0.03run1/iter100/predictions/{patients[i]}'
    # vat_path2 = f'{str1}/vat_305reg1eps0.03run2/iter100/predictions/{patients[i]}'
    # vat_path3 = f'{str1}/vat_305reg1eps0.03run3/iter100/predictions/{patients[i]}'
    #
    # cot_path1 = f'{str1}/cot_205reg_0.05/iter100/predictions/{patients[i]}'
    # cot_path2 = f'{str1}/cot_205reg_0.05_run2/iter100/predictions/{patients[i]}'
    # cot_path3 = f'{str1}/cot_205reg_0.05_run3/iter100/predictions/{patients[i]}'

    # mt_path1 = f'{str1}/MT_2reg_0.05run1/iter100/predictions/{patients[i]}'
    # mt_path2 = f'{str1}/MT_2reg_0.05run2/iter100/predictions/{patients[i]}'
    # mt_path3 = f'{str1}/MT_2reg_0.05run3/iter100/predictions/{patients[i]}'

    # cavat_path1 = f'{str1}/cavatcons_305reg05eps205cons0.05run1/iter099/predictions/{patients[i]}'
    # cavat_path2 = f'{str1}/cavatcons_305reg05eps205cons0.05run2/iter099/predictions/{patients[i]}'
    # cavat_path3 = f'{str1}/cavatcons_305reg05eps205cons0.05run3/iter099/predictions/{patients[i]}'

    cotcavat_path1 = f'{str1}/MTCaVATcons_4reg_701eps206cons_run1/iter100/predictions/{patients[i]}'
    cotcavat_path2 = f'{str1}/MTCaVATcons_4reg_701eps206cons_run2/iter100/predictions/{patients[i]}'
    cotcavat_path3 = f'{str1}/MTCaVATcons_4reg_701eps206cons_run3/iter100/predictions/{patients[i]}'
    #
    # mtcavat_path1 = f'{str1}/cavatcons_301reg0epsRV0.05run1/iter099/predictions/{patients[i]}'
    # mtcavat_path2 = f'{str1}/cavatcons_301reg0epsRV0.05run2/iter099/predictions/{patients[i]}'
    # mtcavat_path3 = f'{str1}/cavatcons_301reg0epsRV0.05run3/iter099/predictions/{patients[i]}'
    #
    # path = [gt_path, f_path1, f_path2, f_path3, p_path1, p_path2, p_path3, Ent_path1, Ent_path2, Ent_path3]
    # path = [gt_path, vat_path1, vat_path2, vat_path3, cot_path1, cot_path2, cot_path3, mt_path1, mt_path2, mt_path3]
    # path = [gt_path, cavat_path1, cavat_path2, cavat_path3, cotcavat_path1, cotcavat_path2, cotcavat_path3, mtcavat_path1, mtcavat_path2, mtcavat_path3]

    path = [gt_path, cotcavat_path1, cotcavat_path2,cotcavat_path3]
    #
    HD_score = ensembel3D(path, i, pxiel_wise=False)
    HD_list.append(HD_score)


list_patient = []
for patient_hd in HD_list:
    method_total = []
    for method_hd in patient_hd:
        for i in range(len(method_hd)):
            method_sub_total = method_hd[i].summary()
        method_total.append(method_sub_total)
    list_patient.append(method_total)


s1_1 = 0
s2_1 = 0
s3_1 = 0
s4_1 = 0
s5_1 = 0
s6_1 = 0
s7_1 = 0
s8_1 = 0
s9_1 = 0


for method in list_patient:

    s1_1 = s1_1 + method[0].get('HD1')
    s2_1 = s2_1 + method[1].get('HD1')
    s3_1 = s3_1 + method[2].get('HD1')
    # s4_1 = s4_1 + method[3].get('HD1')
    # s5_1 = s5_1 + method[4].get('HD1')
    # s6_1 = s6_1 + method[5].get('HD1')
    # s7_1 = s7_1 + method[6].get('HD1')
    # s8_1 = s8_1 + method[7].get('HD1')
    # s9_1 = s9_1 + method[8].get('HD1')


s1_1 = s1_1/len(list_patient)
s2_1 = s2_1/len(list_patient)
s3_1 = s3_1/len(list_patient)
# s4_1 = s4_1/len(list_patient)
# s5_1 = s5_1/len(list_patient)
# s6_1 = s6_1/len(list_patient)
# s7_1 = s7_1/len(list_patient)
# s8_1 = s8_1/len(list_patient)
# s9_1 = s9_1/len(list_patient)

list_final = [s1_1, s2_1, s3_1, s4_1, s5_1, s6_1, s7_1, s8_1, s9_1]
print(list_final)




















