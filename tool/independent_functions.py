import os
import random

import torch
from pathlib import Path
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.io import imsave
from typing import Union, List, Tuple
import matplotlib.pyplot as plt

Tensor = Union[np.ndarray, torch.Tensor]

def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)

def average_list(input_list):
    return sum(input_list) / len(input_list)

def class2one_hot(seg, C, class_dim: int = 1):
    seg = torch.Tensor(seg).type(torch.float)
    # if len(seg.shape) == 3:
    #     seg = seg.unsqueeze(dim=1)
    res = torch.stack([seg == c for c in range(C)], dim=class_dim).type(torch.float)
    return res

def plot_joint_matrix(file_name, joint: Tensor):
    assert joint.dim() == 4, joint.shape
    n1, n2 = joint.shape[0:2]
    fig = plt.figure()
    fig.set_figwidth(15)
    joint = joint.detach().cpu().float().numpy()
    for i1 in range(1, n1 + 1):
        for i2 in range(1, n2 + 1):
            ax = plt.subplot(n1, n2, (i1 - 1) * n1 + i2)
            plt.title(f'{file_name}')

            img = joint[i1 - 1, i2 - 1]
            im_ = ax.imshow(img)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_, cax=cax, orientation='vertical')
    return fig

def plot_seg(img, label):
    fig = plt.figure()
    plt.title(f'{img}')
    gt_volume = tensor2plotable(label)
    plt.imshow(gt_volume, alpha=1, cmap="viridis")
    return fig


class saveimggt:
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self._save_dir: Path = save_dir if isinstance(save_dir, Path) else Path(
            save_dir
        )
        self._save_dir.mkdir(exist_ok=True, parents=True)

    def write(self, img, label, index=0):
        assert img.shape == label.shape
        num = img.shape[0]
        for i in range(num):
            save_img_path = Path(self._save_dir, f"img/symetry_{index}").with_suffix(".png")
            save_gt_path = Path(self._save_dir, f"gt/symetry_{index}").with_suffix(".png")
            save_img_path.parent.mkdir(parents=True, exist_ok=True)
            save_gt_path.parent.mkdir(parents=True, exist_ok=True)

            imsave(str(save_img_path), (img[i]*255).numpy().astype(np.uint8))
            imsave(str(save_gt_path), label[i].numpy().astype(np.uint8))


def _empty_iterator(tensor) -> bool:
    """
    check if a list (tuple) is empty
    """
    from collections.abc import Iterable

    if isinstance(tensor, Iterable):
        if len(tensor) == 0:
            return True
    return False


def _is_tensor(tensor) -> bool:
    """
    return bool indicating if an input is a tensor of numpy or torch.
    """
    if torch.is_tensor(tensor):
        return True
    if isinstance(tensor, np.ndarray):
        return True
    return False


def _is_iterable_tensor(tensor) -> bool:
    """
    return bool indicating if an punt is a list or a tuple of tensor
    """
    from collections.abc import Iterable

    if isinstance(tensor, Iterable):
        if len(tensor) > 0:
            if _is_tensor(tensor[0]):
                return True
    return False


def tensor2plotable(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise TypeError(f"tensor should be an instance of Tensor, given {type(tensor)}")


# below are the functions mostly utilized.
def multi_slice_viewer_debug(
    img_volume: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    *gt_volumes: Tensor,
    no_contour=False,
    block=False,
    alpha=0.2,
    cmap='gray',
) -> None:
    def process_mouse_wheel(event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == "up":
                previous_slice(ax)
            elif event.button == "down":
                next_slice(ax)
        fig.canvas.draw()

    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == "j":
            previous_slice(ax)
        elif event.key == "k":
            next_slice(ax)
        fig.canvas.draw()

    def previous_slice(ax):
        img_volume = ax.img_volume
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(img_volume[ax.index])

        if ax.gt_volume is not None:
            if not no_contour:
                for con in ax.con.collections:
                    con.remove()
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con.remove()
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")

    def next_slice(ax):
        img_volume = ax.img_volume
        ax.index = (
            (ax.index + 1)
            if (ax.index + 1) < img_volume.shape[0]
            else img_volume.shape[0] - 1
        )
        ax.images[0].set_array(img_volume[ax.index])

        if ax.gt_volume is not None:
            if not no_contour:
                for con in ax.con.collections:
                    con.remove()
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con.remove()
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")

    try:
        import matplotlib

        matplotlib.use("tkagg", force=True)
    except Exception as e:
        print(e)

    ## assertion part:
    assert _is_tensor(img_volume) or _is_iterable_tensor(
        img_volume
    ), f"input wrong for img_volume, given {img_volume}."
    assert (
        _is_iterable_tensor(gt_volumes) or gt_volumes == ()
    ), f"input wrong for gt_volumes, given {gt_volumes}."
    if _is_tensor(img_volume):
        img_volume = [img_volume]
    row_num, col_num = len(img_volume), max(len(gt_volumes), 1)

    fig, axs = plt.subplots(row_num, col_num)
    if not isinstance(axs, np.ndarray):
        # lack of numpy wrapper
        axs = np.array([axs])
    axs = axs.reshape((row_num, col_num))

    for _row_num, row_axs in enumerate(axs):
        # each row
        assert len(row_axs) == col_num
        for _col_num, ax in enumerate(row_axs):
            ax.img_volume = tensor2plotable(img_volume[_row_num])
            ax.index = ax.img_volume.shape[0] // 2
            ax.imshow(ax.img_volume[ax.index], cmap=cmap)
            ax.gt_volume = (
                None
                if _empty_iterator(gt_volumes)
                else tensor2plotable(gt_volumes[_col_num])
            )
            try:
                if not no_contour:
                    ax.con = ax.contour(ax.gt_volume[ax.index])
                else:
                    ax.con = ax.imshow(
                        ax.gt_volume[ax.index], alpha=alpha, cmap="viridis"
                    )
            except:
                pass
            ax.axis("off")
            ax.set_title(f"plane = {ax.index}")

    fig.canvas.mpl_connect("key_press_event", process_key)
    fig.canvas.mpl_connect("scroll_event", process_mouse_wheel)
    plt.show(block=block)
