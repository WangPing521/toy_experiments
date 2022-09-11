from deepclustering.dataset.segmentation import ACDCSemiInterface, ProstateSemiInterface
from tool.augment_spleen import train_transformS, val_transformS
from tool.augment import train_transform, val_transform
from deepclustering.utils import tqdm_

from tool.save_images import save_images

dataset_handler = ACDCSemiInterface(unlabeled_data_ratio=0.9, labeled_data_ratio=0.1, seed=1)
# dataset_handler = SpleenSemiInterface(unlabeled_data_ratio=0.9, labeled_data_ratio=0.1, seed=12)
# dataset_handler = ProstateSemiInterface(unlabeled_data_ratio=0.9, labeled_data_ratio=0.1, seed=1)

def get_group_set(dataloader):
    return set(sorted(dataloader.dataset.get_group_list()))


dataset_handler.compile_dataloader_params(
    labeled_batch_size=4,
    unlabeled_batch_size=6,
    val_batch_size=6,
    shuffle=True,
    num_workers=0)

label_loader, unlab_loader, val_loader = dataset_handler.SemiSupervisedDataLoaders(
    labeled_transform=train_transform,
    unlabeled_transform=train_transform,
    val_transform=val_transform,
    group_val=True,
    use_infinite_sampler=True,
)


def val_img_gt(val_loader):

    val_indicator = tqdm_(val_loader)
    for batch_id, data in enumerate(val_indicator):
        image, target, filename = (
            data[0][0].to('cpu'),
            data[0][1].to('cpu'),
            data[1],
        )

        save_images((image*255).squeeze(1), names=filename, root='prostate', mode='img')
        save_images(target.squeeze(1), names=filename, root='prostate', mode='gt')

val_img_gt(val_loader)