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
    subject_num = re.compile(fr"\d+").findall(stem)[0]
    return subject_num


def _glob_folder(current_folder: Path, indicator, postfix: str) -> List[str]:
    return sorted(
        set([extract_suject_num(x, indicator) for x in current_folder.rglob(f"*.{postfix}")])
    )


def ensembel3D(path):
    volum = []
    for img_path in path:
        indicator = f'Case_0_'
        img_list = _glob_folder(Path(img_path), indicator, 'png')
        group_list = []
        for i in range(len(img_list)):
            group_sublist = []
            for file_png in os.listdir(img_path):
                if file_png[4:6] == img_list[i]:
                    group_sublist.append(file_png)
            group_list.append(group_sublist)

        vol = []
        for img_subfile in group_list:
            vol_sub = []
            sub_num = []
            for img_file in img_subfile:
                sub_num.append(img_file[9:11])
            num_list = sorted(sub_num)
            for img_num in range(len(img_subfile)):
                img_i = Image.open(f'{img_path}/{img_subfile[img_num][:-6]}{num_list[img_num]}.png')
                img_slice = torch.Tensor(np.array(img_i))
                vol_sub.append(img_slice.unsqueeze(0))
            vol.append(torch.cat(vol_sub, dim=0))
        volum.append(vol)
    return volum


trans = transforms.ToTensor()
strs = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/promise_val'
strs1 = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/Promise12_0.08'

patients = ['02', '16', '19', '24', '25', '26', '30', '31', '32', '48']
for i in range(len(patients)):
    img = f'{strs}/img/{patients[i]}'
    gt_path = f'{strs}/gt/{patients[i]}'
    # f1 = f'{strs1}/baseline_f_prostate_1Trun1/iter100/predictions/{patients[i]}'
    # f2 = f'{strs1}/baseline_f_prostate_1Trun2/iter100/predictions/{patients[i]}'
    # f3 = f'{strs1}/baseline_f_prostate_1Trun3/iter100/predictions/{patients[i]}'
    # p1 = f'{strs1}/baseline_p_prostate_1Trun1/iter100/predictions/{patients[i]}'
    # p2 = f'{strs1}/baseline_p_prostate_1Trun2/iter100/predictions/{patients[i]}'
    # p3 = f'{strs1}/baseline_p_prostate_1Trun3/iter100/predictions/{patients[i]}'
    # Ent1 = f'{strs1}/EntMin_prostate_301reg05Trun1/iter100/predictions/{patients[i]}'
    # Ent2 = f'{strs1}/EntMin_prostate_301reg05Trun2/iter100/predictions/{patients[i]}'
    # Ent3 = f'{strs1}/EntMin_prostate_301reg05Trun3/iter100/predictions/{patients[i]}'
    vat1 = f'{strs1}/vat_prostate_305reg05eps05Trun1/iter100/predictions/{patients[i]}'
    vat2 = f'{strs1}/vat_prostate_305reg05eps05Trun2/iter100/predictions/{patients[i]}'
    # vat3 = f'{strs1}/vat_prostate_301reg01eps05Trun3/iter100/predictions/{patients[i]}'
    # cot1 = f'{strs1}/cot_307reg_05T205cons/iter099/predictions/{patients[i]}'
    # cot2 = f'{strs1}/EntMin_prostate_501reg10Trun1/iter100/predictions/{patients[i]}'
    # cot3 = f'{strs1}/EntMin_prostate_501reg10Trun1/iter100/predictions/{patients[i]}'
    # mt1 = f'{strs1}/MT_prostate_4reg_run1/iter099/predictions/{patients[i]}'
    # mt2 = f'{strs1}/EntMin_prostate_501reg10Trun1/iter100/predictions/{patients[i]}'
    # mt3 = f'{strs1}/EntMin_prostate_501reg10Trun1/iter100/predictions/{patients[i]}'
    cavat1 = f'{strs1}/cavatcons_205reg1eps305cons/iter099/predictions/{patients[i]}'
    cavat2 = f'{strs1}/cavatcons_205reg1eps305cons_run2/iter099/predictions/{patients[i]}'
    cavat3 = f'{strs1}/cavatcons_205reg1eps305cons_run3/iter099/predictions/{patients[i]}'
    # cotCaVAT1 = f'{strs1}/cotCaVATcons_prostate_307reg_401eps203cons_run1/iter100/predictions/{patients[i]}'
    # cotCaVAT2 = f'{strs1}/cotCaVATcons_prostate_307reg_401eps301cons_run2/iter099/predictions/{patients[i]}'
    # cotCaVAT3 = f'{strs1}/cotCaVATcons_prostate_307reg_401eps301cons_run3/iter099/predictions/{patients[i]}'
    # mtCaVAT1 = f'{strs1}/MTCaVAT_prostate_4reg_601eps305cons_run1/iter099/predictions/{patients[i]}'
    # mtCaVAT2 = f'{strs1}/MTCaVAT_prostate_4reg_601eps305cons_run2/iter099/predictions/{patients[i]}'
    # mtCaVAT3 = f'{strs1}/MTCaVAT_prostate_4reg_601eps305cons_run3/iter099/predictions/{patients[i]}'
    path = [img, gt_path, vat1, vat2, cavat1, cavat2, cavat3]
    volum = ensembel3D(path)
    print('Next patient...')
    multi_slice_viewer_debug([volum[0][0]], volum[1][0], volum[2][0], volum[3][0], volum[4][0], volum[5][0], volum[6][0])

# spleen
# patients = ['_08', '_10', '_14', '_20', '_31']
# p_idx = 4
# num = '18'
# img = Image.open(f'{strs}/spleen/img/{patients[p_idx]}/Patient{patients[p_idx]}_0{num}.png')
# gt  = Image.open(f'{strs}/spleen/gt/{patients[p_idx]}/Patient{patients[p_idx]}_0{num}.png')
# run1 = Image.open(f'{strs}/uncertaintyMT_S/uncertainty_s/0.1/Spleen_uncertainty_MT_run1/iter100/prediction/{patients[p_idx]}/Patient{patients[p_idx]}_0{num}.png')
# run2 = Image.open(f'{strs}/uncertaintyMT_S/uncertainty_s/0.1/Spleen_uncertainty_MT_run1/iter100/prediction/{patients[p_idx]}/Patient{patients[p_idx]}_0{num}.png')
# run3 = Image.open(f'{strs}/uncertaintyMT_S/uncertainty_s/0.1/Spleen_uncertainty_MT_run1/iter100/prediction/{patients[p_idx]}/Patient{patients[p_idx]}_0{num}.png')
# multi_slice_viewer_debug([torch.flip(trans(img), [0, 2])], torch.flip(trans(gt), [0, 2]), torch.flip(trans(run1), [0, 2]), torch.flip(trans(run2), [0, 2]), torch.flip(trans(run3), [0, 2]))


# img1 = Image.open(f'entropy/entropy_rv/r.png')
# img11 = torch.Tensor(np.array(img1))
# plt.imshow(trans(img1).squeeze(0), cmap=plt.get_cmap('viridis'))
# plt.show()
# print('ok')
