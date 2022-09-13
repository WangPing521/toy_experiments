from PIL import Image
from deepclustering2.viewer import multi_slice_viewer_debug
from torchvision import transforms
import os


trans = transforms.ToTensor()
val_str = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/acdc_val'
pred_strs = '../../../Daily_Research/TMI_constraint/PaperResults/visualization/LV_convexity_003'

img_list = os.listdir(pred_strs)

for img in img_list:
    img =  Image.open(f'{val_str}/img/{img}')
    gt = Image.open(f'{val_str}/gt/{img}')
    pred = Image.open(f'{pred_strs}/prediction/{img}')

    img = trans(img)
    gt = trans(gt)
    pred = trans(pred)

    multi_slice_viewer_debug([img], gt, pred)

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

