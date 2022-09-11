from nilearn.image import resample_to_img
from medpy.io import load
import nibabel as nb
import numpy as np
from scipy.ndimage import zoom
import torch

from tool.independent_functions import class2one_hot


def reviseResolution(FLAIR, label, x, y, z):
    FLAIR_1 = zoom(FLAIR, (x, y, z))
    labels_onehot = class2one_hot(label, 9)  # 0,1,2,3,7,8
    labels_onehot = labels_onehot.squeeze(0)
    labels_origin = np.concatenate(
        (torch.Tensor(labels_onehot[0]).unsqueeze(0),
         torch.Tensor(labels_onehot[1]).unsqueeze(0),
         torch.Tensor(labels_onehot[2]).unsqueeze(0),
         torch.Tensor(labels_onehot[3]).unsqueeze(0),
         torch.Tensor(labels_onehot[4]).unsqueeze(0),
         torch.Tensor(labels_onehot[5]).unsqueeze(0),
         torch.Tensor(labels_onehot[6]).unsqueeze(0),
         torch.Tensor(labels_onehot[7]).unsqueeze(0),
         torch.Tensor(labels_onehot[8]).unsqueeze(0),
         ), axis=0)
    label_1_origin = torch.Tensor(labels_origin.transpose(0, 3, 1, 2)).unsqueeze(0).max(1)[1].squeeze(0)
    labels_onehot1 = torch.Tensor(zoom(labels_onehot[0], (x, y, z), order=0))
    labels_onehot2 = torch.Tensor(zoom(labels_onehot[1], (x, y, z), order=0))
    labels_onehot3 = torch.Tensor(zoom(labels_onehot[2], (x, y, z), order=0))
    labels_onehot4 = torch.Tensor(zoom(labels_onehot[3], (x, y, z), order=0))
    labels_onehot5 = torch.Tensor(zoom(labels_onehot[4], (x, y, z), order=0))
    labels_onehot6 = torch.Tensor(zoom(labels_onehot[5], (x, y, z), order=0))
    labels_onehot7 = torch.Tensor(zoom(labels_onehot[6], (x, y, z), order=0))
    labels_onehot8 = torch.Tensor(zoom(labels_onehot[7], (x, y, z), order=0))
    labels_onehot9 = torch.Tensor(zoom(labels_onehot[8], (x, y, z), order=0))

    labels = np.concatenate(
        (labels_onehot1.unsqueeze(0),
         labels_onehot2.unsqueeze(0),
         labels_onehot3.unsqueeze(0),
         labels_onehot4.unsqueeze(0),
         labels_onehot5.unsqueeze(0),
         labels_onehot6.unsqueeze(0),
         labels_onehot7.unsqueeze(0),
         labels_onehot8.unsqueeze(0),
         labels_onehot9.unsqueeze(0),
         ), axis=0
    )
    label_1 = np.float64(torch.Tensor(labels).unsqueeze(0).max(1)[1].squeeze(0))
    return FLAIR_1, label_1


template = nb.load('dataset/iSeg2017/iSeg-2017-Training/subject-1-T1.nii')
mrbrain = nb.load('dataset/training/1/pre/FLAIR.nii.gz')
mrbrain_label = nb.load('dataset/training/1/segm.nii.gz')

mrbrain_images, img_header = load('dataset/training/1/pre/FLAIR.nii.gz')
mrbrain_target, target_header = load('dataset/training/1/segm.nii.gz')

#  original MRbrain
mrbrain_img = mrbrain_images.astype(np.float32)
mrbrain_t = mrbrain_target.astype(np.float32)

w, h, d, _ = template.get_data().shape
w1, h1, d1 = mrbrain_img.shape

# resized MRbrain via zoom
mrbrain_imgr, mrbrain_tr = reviseResolution(mrbrain_img, mrbrain_t, x=w/w1, y=h/h1, z=d/d1)

# resized MRbrain via nilean
resampled_mrbrain_img = resample_to_img(mrbrain, template)
resampled_mrbrain_label = resample_to_img(mrbrain_label, template)




