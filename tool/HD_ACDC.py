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


strs = '../../../../PHD_documents/papers_work/Medical_Image_Analysis/TMISpecial_200802-201116/major_revision/new_experiments/uncertaintyMT/acdc'
patients = ['006', '013', '018', '019', '028', '033', '034', '037', '039', '040', '046',
            '050', '052', '054', '062', '063', '065', '066', '069', '075', '078', '080',
            '093', '098']
index1 = ['01', '06', '09', '10', '11', '12', '13', '14']
C = 4

_meter_interface = MeterInterface()

for kk in range(20):
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
            vol_sub.append(torch.cat(vol, dim=0))
        volum.append(vol_sub)
    list_methods = []
    for method_i in range(len(volum)-1):
        list_sub = []
        for group_i in range(len(volum[method_i])):
            gt = volum[0][group_i]
            if volum[method_i+1][group_i].unique().sum() != 0 and gt.unique().sum() != 0:
                onehot_pred, onehot_target = _convert2onehot(volum[method_i+1][group_i], gt, C)
                _meter_interface[f"hd{method_i}"].add(onehot_pred.transpose(1, 0).unsqueeze(0), onehot_target.transpose(1, 0).unsqueeze(0))
                list_sub.append(_meter_interface[f"hd{method_i}"].value()[0])
        list_methods.append(list_sub)
    return list_methods


# run1
str1 = '../../../../PHD_documents/papers_work/CaVAT_TMI/segmentations/ACDC_0.05'

HD_list = []
for i in range(len(patients)):
    gt_path = f'{strs}/gt/{patients[i]}'
    f1 = f'{str1}/baseline_f_run1/iter099/predictions/{patients[i]}'
    f2 = f'{str1}/baseline_f_run2/iter099/predictions/{patients[i]}'
    f3 = f'{str1}/baseline_f_run3/iter099/predictions/{patients[i]}'

    p1 = f'{str1}/baseline_p_run1/iter099/predictions/{patients[i]}'
    p2 = f'{str1}/baseline_p_run2/iter099/predictions/{patients[i]}'
    p3 = f'{str1}/baseline_p_run3/iter099/predictions/{patients[i]}'

    Ent1 = f'{str1}/EntMin_305reg05Trun1/iter099/predictions/{patients[i]}'
    Ent2 = f'{str1}/EntMin_305reg05Trun2/iter099/predictions/{patients[i]}'
    Ent3 = f'{str1}/EntMin_305reg05Trun3/iter099/predictions/{patients[i]}'

    vat1 = f'{str1}/vat_405reg05eps05Trun1/iter099/predictions/{patients[i]}'
    vat2 = f'{str1}/vat_405reg05eps05Trun2/iter099/predictions/{patients[i]}'
    vat3 = f'{str1}/vat_405reg05eps05Trun3/iter099/predictions/{patients[i]}'

    MT1 = f'{str1}/MT_prostate_2reg_run1/iter099/predictions/{patients[i]}'
    MT2 = f'{str1}/MT_prostate_2reg_run2/iter099/predictions/{patients[i]}'
    MT3 = f'{str1}/MT_prostate_2reg_run3/iter099/predictions/{patients[i]}'

    # path = [gt_path, f1, f2, f3, p1, p2, p3, Ent1, Ent2, Ent3]
    path = [gt_path,
            f1, f2, f3,
            p1, p2, p3,
            Ent1, Ent2, Ent3,
            vat1, vat2, vat3,
            MT1, MT2, MT3,
            ]
    HD_score = ensembel3D(path, i, pxiel_wise=False)
    HD_list.append(HD_score)


s1_1,  s1_2, s1_3 = 0, 0, 0
s2_1,  s2_2, s2_3 = 0, 0, 0
s3_1,  s3_2, s3_3 = 0, 0, 0
s4_1,  s4_2, s4_3 = 0, 0, 0
s5_1,  s5_2, s5_3 = 0, 0, 0
s6_1,  s6_2, s6_3 = 0, 0, 0
s7_1,  s7_2, s7_3 = 0, 0, 0
s8_1,  s8_2, s8_3 = 0, 0, 0
s9_1,  s9_2, s9_3 = 0, 0, 0
s10_1, s10_2, s10_3 = 0, 0, 0
s11_1, s11_2, s11_3 = 0, 0, 0
s12_1, s12_2, s12_3 = 0, 0, 0
s13_1, s13_2, s13_3 = 0, 0, 0
s14_1, s14_2, s14_3 = 0, 0, 0
s15_1, s15_2, s15_3 = 0, 0, 0


for method in HD_list:
    for i in range(len(method[0])):
        s1_1 = s1_1 + method[0][i][0]
        s1_2 = s1_2 + method[0][i][1]
        s1_3 = s1_3 + method[0][i][2]

        s2_1 = s2_1 + method[1][i][0]
        s2_2 = s2_2 + method[1][i][1]
        s2_3 = s2_3 + method[1][i][2]
        # #
        s3_1 = s3_1 + method[2][i][0]
        s3_2 = s3_2 + method[2][i][1]
        s3_3 = s3_3 + method[2][i][2]
        # #
        s4_1 = s4_1 + method[3][i][0]
        s4_2 = s4_2 + method[3][i][1]
        s4_3 = s4_3 + method[3][i][2]
        #
        s5_1 = s5_1 + method[4][i][0]
        s5_2 = s5_2 + method[4][i][1]
        s5_3 = s5_3 + method[4][i][2]
        #
        s6_1 = s6_1 + method[5][i][0]
        s6_2 = s6_2 + method[5][i][1]
        s6_3 = s6_3 + method[5][i][2]

        s7_1 = s7_1 + method[6][i][0]
        s7_2 = s7_2 + method[6][i][1]
        s7_3 = s7_3 + method[6][i][2]
        #
        s8_1 = s8_1 + method[7][i][0]
        s8_2 = s8_2 + method[7][i][1]
        s8_3 = s8_3 + method[7][i][2]
        #
        s9_1 = s9_1 + method[8][i][0]
        s9_2 = s9_2 + method[8][i][1]
        s9_3 = s9_3 + method[8][i][2]

        s10_1 = s10_1 + method[9][i][0]
        s10_2 = s10_2 + method[9][i][1]
        s10_3 = s10_3 + method[9][i][2]

        s11_1 = s11_1 + method[10][i][0]
        s11_2 = s11_2 + method[10][i][1]
        s11_3 = s11_3 + method[10][i][2]

        s12_1 = s12_1 + method[11][i][0]
        s12_2 = s12_2 + method[11][i][1]
        s12_3 = s12_3 + method[11][i][2]

        s13_1 = s13_1 + method[12][i][0]
        s13_2 = s13_2 + method[12][i][1]
        s13_3 = s13_3 + method[12][i][2]

        s14_1 = s14_1 + method[13][i][0]
        s14_2 = s14_2 + method[13][i][1]
        s14_3 = s14_3 + method[13][i][2]

        s15_1 = s15_1 + method[14][i][0]
        s15_2 = s15_2 + method[14][i][1]
        s15_3 = s15_3 + method[14][i][2]

count = 0
for i in range(len(HD_list)):
    count = count + HD_list[i][0].__len__()

s1_1 = s1_1/count
s1_2 = s1_2/count
s1_3 = s1_3/count

s2_1 = s2_1/count
s2_2 = s2_2/count
s2_3 = s2_3/count
#
s3_1 = s3_1/count
s3_2 = s3_2/count
s3_3 = s3_3/count
#
s4_1 = s4_1/len(HD_list)
s4_2 = s4_2/len(HD_list)
s4_3 = s4_3/len(HD_list)
#
s5_1 = s5_1/len(HD_list)
s5_2 = s5_2/len(HD_list)
s5_3 = s5_3/len(HD_list)
#
s6_1 = s6_1/len(HD_list)
s6_2 = s6_2/len(HD_list)
s6_3 = s6_3/len(HD_list)

s7_1 = s7_1/len(HD_list)
s7_2 = s7_2/len(HD_list)
s7_3 = s7_3/len(HD_list)
#
s8_1 = s8_1/len(HD_list)
s8_2 = s8_2/len(HD_list)
s8_3 = s8_3/len(HD_list)
#
s9_1 = s9_1/len(HD_list)
s9_2 = s9_2/len(HD_list)
s9_3 = s9_3/len(HD_list)
#
s10_1 = s10_1/len(HD_list)
s10_2 = s10_2/len(HD_list)
s10_3 = s10_3/len(HD_list)
#
s11_1 = s11_1/len(HD_list)
s11_2 = s11_2/len(HD_list)
s11_3 = s11_3/len(HD_list)
#
s12_1 = s12_1/len(HD_list)
s12_2 = s12_2/len(HD_list)
s12_3 = s12_3/len(HD_list)

s13_1 = s13_1/len(HD_list)
s13_2 = s13_2/len(HD_list)
s13_3 = s13_3/len(HD_list)
#
s14_1 = s14_1/len(HD_list)
s14_2 = s14_2/len(HD_list)
s14_3 = s14_3/len(HD_list)
#
s15_1 = s15_1/len(HD_list)
s15_2 = s15_2/len(HD_list)
s15_3 = s15_3/len(HD_list)

list_final = [s1_1, s1_2, s1_3,
              s2_1, s2_2, s2_3,
              s3_1, s3_2, s3_3,
              s4_1, s4_2, s4_3,
              s5_1, s5_2, s5_3,
              s6_1, s6_2, s6_3,
              s7_1, s7_2, s7_3,
              s8_1, s8_2, s8_3,
              s9_1, s9_2, s9_3,
              s10_1, s10_2, s10_3,
              s11_1, s11_2, s11_3,
              s12_1, s12_2, s12_3,
              s13_1, s13_2, s13_3,
              s14_1, s14_2, s14_3,
              s15_1, s15_2, s15_3,
              
              ]
print(list_final)




















