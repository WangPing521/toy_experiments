import os
from torch.utils.data import Dataset
from torchvision.io import read_image


class DataloaderIter:
    def __init__(self, dataloader) ->None:
        self._dataloader = dataloader
        self._dataloader_iter = iter(dataloader)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return self._dataloader_iter.__next__()
        except:
            self._dataloader_iter = iter(self._dataloader)
            return self._dataloader_iter.__next__()


class SymetryData(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, ratio=0.1, indicator='val'):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.data_container = os.listdir(f'{self.img_dir}/img')
        self.indicator = indicator
        self.ratio = ratio
        if self.indicator in ['lab_train', 'unlab_train']:
            num = len(self.data_container)
            self.lab_container = self.data_container[0:round(num * self.ratio)]
            self.unlab_container = self.data_container[round(num * self.ratio):]


    def __len__(self):
        if self.indicator in ['val']:
            return len(self.data_container)
        elif self.indicator in ['lab_train']:
            return len(self.lab_container)
        elif self.indicator in ['unlab_train']:
            return len(self.unlab_container)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'img')
        gt_path = os.path.join(self.img_dir, 'gt')

        if self.indicator in ['val']:
            image = read_image(os.path.join(img_path, self.data_container[idx]))
            label = read_image(os.path.join(gt_path, self.data_container[idx]))
            filename = self.data_container[idx]
        elif self.indicator in ['lab_train']:
            image = read_image(os.path.join(img_path, self.lab_container[idx]))
            label = read_image(os.path.join(gt_path, self.lab_container[idx]))
            filename = self.lab_container[idx]
        elif self.indicator in ['unlab_train']:
            image = read_image(os.path.join(img_path, self.unlab_container[idx]))
            label = read_image(os.path.join(gt_path, self.unlab_container[idx]))
            filename = self.unlab_container[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image / 255, label, filename




