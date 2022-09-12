import torch
from skimage.morphology import flood_fill
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import cv2
import numpy as np

from tool.independent_functions import dscintersaction, average_list, dscunion, class2one_hot

device = 'cuda'
def local_cons_binary_convex(sample_list, scale):
    kernel = torch.ones(1, 1, 3, 3)
    kernel = torch.FloatTensor(kernel)
    weight = nn.Parameter(data=kernel, requires_grad=False)

    patch = torch.ones(1, 1, scale, scale)
    patch = torch.FloatTensor(patch)
    patch_weight = nn.Parameter(data=patch, requires_grad=False)

    reward_samples = []
    for samples_img in sample_list:

        # find center of the flood fill and get one connected component
        count_neighbor = F.conv2d(samples_img, weight, padding=1)
        count_neighbors = count_neighbor * samples_img
        m = count_neighbors.view(count_neighbors.shape[0], -1).argmax(1)
        max_index = torch.cat(((m // count_neighbors.shape[2]).view(-1, 1), (m % count_neighbors.shape[2]).view(-1, 1)), dim=1)
        for i in range(samples_img.shape[0]):
            if i == 0:
                fill_connect = torch.Tensor(flood_fill(samples_img[i].squeeze(0).numpy(), (max_index[:, 0][i], max_index[:, 1][i]), 9)).unsqueeze(0).unsqueeze(0)
            else:
                fill_connect_img = torch.Tensor(flood_fill(samples_img[i].squeeze(0).numpy(), (max_index[:, 0][i], max_index[:, 1][i]), 9))
                fill_connect = torch.cat([fill_connect, fill_connect_img.unsqueeze(0).unsqueeze(0)], dim=1)
        fill_connect = torch.where(fill_connect == 9, torch.Tensor([1]), torch.Tensor([0])).transpose(1, 0)

        # compute local reward
        F_constraint = F.conv2d(fill_connect, patch_weight, padding=int((patch.shape[2] - 1) / 2))
        S_constraint = F.conv2d(samples_img, patch_weight, padding=int((patch.shape[2] - 1) / 2))

        F_foregroudneigbors = F_constraint * fill_connect
        S_foregroudneigbors = S_constraint * samples_img

        # case 1: binary---R(0, 1)
        pixel_reward = (F_foregroudneigbors == S_foregroudneigbors).float()

        # case 2: binary---R(-1, 1)
        # pixel_reward1 = (F_foregroudneigbors == S_foregroudneigbors).float()
        # pixel_reward = torch.where(torch.Tensor(pixel_reward1) == 0, torch.Tensor([-1]), torch.Tensor(pixel_reward1))

        # case 3: minus----R(negative, 0)
        # pixel_reward = (S_foregroudneigbors - F_foregroudneigbors).float()

        reward_samples.append(pixel_reward)
    return reward_samples

def ContourEstimator(x:Tensor):
    for i in range(x.shape[0]):
        contours, hierarchy = cv2.findContours(x[i].squeeze(0).cpu().numpy().astype(dtype=np.uint8), 0, 1)
        regions = []
        for c in contours:
            regions.append(cv2.contourArea(c))
        convex_contour = (torch.zeros_like(x[i].squeeze(0))).cpu().numpy().astype(dtype=np.float32)
        try:
            max_id = np.argsort(-np.array(regions))[0]
            # cv2.fillConvexPoly(convex_contour, cnt_max, (255, 0, 255))
            convex_contour = cv2.drawContours(convex_contour, contours, max_id, 1, cv2.FILLED)
            convex_contour = (torch.Tensor(convex_contour))
        except:
            print('no contour and no hull.')

        if type(convex_contour) is np.ndarray:
            convex_contour = torch.Tensor(convex_contour)
        if i == 0:
            convex_contours = convex_contour.unsqueeze(0)
        else:
            convex_contours = torch.cat([convex_contours, convex_contour.unsqueeze(0)], 0)
    return convex_contours


def symetry_reward(sample_list, reward_type='hard'): # len(sample_list) = 4
    symmetry_errors_list, all_shapes_list, fg_contour_list = [], [], []

    for samples_img in sample_list: # samples_img: [3,1,256,256]
        fg_contour = ContourEstimator(samples_img)
        #todo: estimate symetry
        contour_index = torch.where(fg_contour==1)
        num_samples = samples_img.shape[0]

        for i in range(num_samples):
            all_shape_tmp = torch.ones_like(fg_contour[i]) * fg_contour[i]
            sample_contouridx = torch.where(contour_index[0]==i)
            center_position = torch.floor(contour_index[2][sample_contouridx].float().mean()) # center

            # center_line = [center_position-10, center_position-5, center_position-3, center_position, center_position+3, center_position+5, center_position+10]
            center_line = [center_position]
            tmp = 65536
            all_shape = torch.zeros_like(samples_img[i].squeeze(0))
            symmetry_error = torch.zeros_like(samples_img[i].squeeze(0))

            for center_position_unk in center_line:
                yy = 2 * center_position_unk - contour_index[2][min(sample_contouridx[0]):max(sample_contouridx[0])+1]
                yy = torch.where(yy > 255, torch.Tensor([255.]), yy)

                all_shape_tmp[contour_index[1][sample_contouridx], yy.long()] = 1
                symmetry_error_tmp = all_shape_tmp - fg_contour[i]
                select_center = symmetry_error_tmp.sum()

                if select_center < tmp:
                    all_shape = all_shape_tmp
                    symmetry_error = symmetry_error_tmp
                    tmp = select_center

            if i == 0:
                all_shapes = all_shape.unsqueeze(0)
                symmetry_errors = symmetry_error.unsqueeze(0)
            else:
                all_shapes = torch.cat([all_shapes, all_shape.unsqueeze(0)], dim=0)
                symmetry_errors = torch.cat([symmetry_errors, symmetry_error.unsqueeze(0)], dim=0)
        all_shapes_list.append(all_shapes)
        symmetry_errors_list.append(symmetry_errors)
        fg_contour_list.append(fg_contour)
    return all_shapes_list, symmetry_errors_list, fg_contour_list

def symmetry_error(pred, label):
    assert pred.shape == label.shape
    batch_num = pred.shape[0]
    fg_contour = ContourEstimator(pred)
    contour_index = torch.where(fg_contour == 1)
    error_list, dsc_list= [], []
    for i in range(batch_num):
        all_shape = torch.ones_like(fg_contour[i]) * fg_contour[i]
        sample_contouridx = torch.where(contour_index[0] == i)
        center_position = torch.floor(contour_index[2][sample_contouridx].float().mean())  # center

        yy = 2 * center_position - contour_index[2][min(sample_contouridx[0]):max(sample_contouridx[0]) + 1]
        yy = torch.where(yy > 255, torch.Tensor([255.]), yy)
        all_shape[contour_index[1][sample_contouridx], yy.long()] = 1
        symmetry_error_tmp = all_shape - fg_contour[i]
        symmetry_error = symmetry_error_tmp.sum()

        error = symmetry_error / all_shape.sum()
        error_list.append(error)

        onehot_pred = class2one_hot(pred.squeeze(1).cpu(), C=2).to(device).to(device)
        onehot_target = class2one_hot(label.squeeze(1).cpu(), C=2).to(device).to(device)

        interaction, union = (
            dscintersaction(onehot_pred, onehot_target),
            dscunion(onehot_pred, onehot_target),
        )
        dice = (2 * interaction.sum(0) + 1e-6) / (union.sum(0) + 1e-6)
        dsc_list.append(dice)

    symmetry_error = average_list(error_list)
    dice = average_list(dsc_list)

    return dice, symmetry_error


