from PIL import Image
from deepclustering2.viewer import multi_slice_viewer_debug
from torchvision import transforms
from deepclustering2.utils import Path
from typing import Union, Tuple, List
import re
import os
import torch
import numpy as np


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


def ensembel3D(path, k):
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
                vol.append(img_slice.unsqueeze(0))
            vol_sub.append(torch.cat(vol, dim=0))  # elements in vol_sub are 3D volume

        if img_path.find('/gt/') != -1:
            fg_bgs = []
            for gt_i in range(len(vol_sub)):
                # LV
                fg_bg = torch.where(vol_sub[gt_i] == 3, torch.Tensor([1]), torch.Tensor([0]))
                fg_bgs.append(fg_bg)
            assert fg_bgs.__len__() == vol_sub.__len__()
            vol_sub = fg_bgs

        volum.append(vol_sub)  # volum: all 3D volumes for all methods
    return volum


trans = transforms.ToTensor()
strs = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/acdc_val'
strs1 = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/LV_convexity_003'

# acdc
patients = ['006', '013', '018', '019', '028', '033', '034', '037', '039', '040', '046',
            '050', '052', '054', '062', '063', '065', '066', '069', '075', '078', '080',
            '093', '098']

for i in range(len(patients)):
    img = f'{strs}/img/{patients[i]}'
    gt_path = f'{strs}/gt/{patients[i]}'

    # p1 = f'{strs1}/baseline_p_run1/iter099/predictions/{patients[i]}'
    # Ent1 = f'{strs1}/EntMin_301reg05Trun1/iter099/predictions/{patients[i]}'

    vat1 = f'{strs1}/vat_301reg_run1/iter099/predictions/{patients[i]}'
    vat2 = f'{strs1}/vat_301reg_run2/iter099/predictions/{patients[i]}'

    cavat0 = f'{strs1}/cavat_301reg_205cons_run1/iter099/predictions/{patients[i]}'
    cavat1 = f'{strs1}/cavat_301reg_205cons_run2/iter099/predictions/{patients[i]}'
    cavat2 = f'{strs1}/cavat_301reg_205cons_run3/iter099/predictions/{patients[i]}'

    path = [img, gt_path, vat1, vat2, cavat0, cavat1, cavat2]
    volum = ensembel3D(path, i)
    ks = len(volum[0])

    for k in range(ks):
        print(f'{i}patient, {k}th scan:')
        multi_slice_viewer_debug([volum[0][k]], volum[1][k], volum[2][k], volum[3][k],
                             volum[4][k], volum[5][k], volum[6][k])
    print('Next patient...')

# LV Myo RV
# target = trans(gt)
# # fg = Image.open(f'{strs}/acdc/gt/{patients[2]}/patient{patients[2]}_{index1[3]}_0_0{5}.png')
# # fg_lab = trans(fg)
# # LV
# # target = torch.where(target == fg_lab.unique()[3], torch.Tensor([1]), torch.Tensor([0]))
# # Myo
# # target = torch.where(target == fg_lab.unique()[2], torch.Tensor([1]), torch.Tensor([0]))
# # RV
# # target = torch.where(target == fg_lab.unique()[1], torch.Tensor([1]), torch.Tensor([0]))

