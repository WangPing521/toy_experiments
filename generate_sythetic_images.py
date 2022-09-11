import os
from PIL import Image
from numpy.random import normal
from torchvision import transforms
from tool.independent_functions import class2one_hot, saveimggt, multi_slice_viewer_debug
import torch

trans = transforms.ToTensor()
def generation(pilImage, toyclasses=2):
    generator = trans(pilImage)
    h, w = generator[0].shape
    targetgenerator = generator[0]

    if toyclasses == 3:
        tmpmap = torch.where(targetgenerator < 0.3, torch.Tensor([10]), targetgenerator)
        tmpmap = torch.where(tmpmap < 0.7, torch.Tensor([11]), tmpmap)
        tmpmap = torch.where(tmpmap < 1, torch.Tensor([12]), tmpmap)

        targetmap = torch.where(tmpmap == 10, torch.Tensor([0]), tmpmap)
        targetmap = torch.where(tmpmap == 11, torch.Tensor([1]), targetmap)
        targetmap = torch.where(tmpmap == 12, torch.Tensor([2]), targetmap)

        mu = [70, 50, 30]
        sigma = [5, 3, 3]

    elif toyclasses == 2:
        targetmap = torch.where(targetgenerator < 0.8, torch.Tensor([1]), torch.Tensor([0]))
        mu = [0.2, 0.5]
        sigma = [0.2, 0.5]

    size = h * w
    assert len(mu) == len(sigma)

    classlist = []
    for i in range(len(mu)):
        sample = normal(mu[i], sigma[i], size)
        sample = torch.Tensor(sample)
        sampleClass = sample.view(h, w)
        classlist.append(sampleClass)

    # generate images
    onehot_target = class2one_hot(targetmap.unsqueeze(0), toyclasses).squeeze(2)
    onehot_target = onehot_target.squeeze(0)
    img = torch.zeros_like(targetmap)
    for i in range(len(classlist)):
        img = img + onehot_target[i] * classlist[i]

    return img.unsqueeze(0), onehot_target


if __name__ == '__main__':

    data_root = 'non_symetry/img'
    img_list = os.listdir(data_root)
    index = 0
    for img in img_list:
        index = index + 1
        generatorbase = Image.open(f'{data_root}/{img}')
        img_s, onehot_target = generation(generatorbase, toyclasses=2)
        target = onehot_target.unsqueeze(0).max(1)[1]

        # multi_slice_viewer_debug([img_s], target)
        # multi_slice_viewer_debug([img_s], target, no_contour=True)

        writer = saveimggt(save_dir='dataset/nonsymetry_images')
        writer.write(img_s, label=target, index=index)



