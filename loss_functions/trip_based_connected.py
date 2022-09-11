import numpy as np
import torch
from torch.distributions import Bernoulli
from utils import average_list


def global_connect(binary_map):
    rows, cols = binary_map.shape
    groups = []
    label = 0
    # collect all the groups in each row
    for row in range(rows):
        groups_row = []
        col = 0
        while col < cols:
            if binary_map[row][col] == 1:
                start_idx = col
                if start_idx == cols-1:
                    label = label + 1
                    groups_row.append((start_idx, start_idx, label))
                    break
                for idx in range(1, cols-col):
                    if binary_map[row][col+idx] == 0:
                        end_idx = max(0, col + idx - 1)
                        label = label + 1
                        groups_row.append((start_idx, end_idx, label))
                        col = col + idx + 1
                        break
                    if col+idx == cols-1 and binary_map[row][col+idx] == 1:
                        end_idx = cols-1
                        label = label + 1
                        groups_row.append((start_idx, end_idx, label))
                        col = end_idx + 1
                        break
            else:
                col = col + 1
        groups.append(groups_row)

    # collect all equal pairs
    pairs = []
    for row_idx in range(1, len(groups)):
        one_num = len(groups[row_idx-1])
        two_num = len(groups[row_idx])
        for one_idx in range(one_num):
            for two_idx in range(two_num):
                if groups[row_idx][two_idx][1] < groups[row_idx-1][one_idx][0]-1 or groups[row_idx-1][one_idx][1] < groups[row_idx][two_idx][0]-1:
                    continue
                else:
                    pairs.append((groups[row_idx-1][one_idx][2], groups[row_idx][two_idx][2]))
    try:
        num_componets = equalgroup(pairs, label)
    except:
        num_componets = label

    return num_componets


def equalgroup(pairs, max_labels):
    maxLabel = np.max(pairs)
    eqTab = np.zeros((maxLabel, maxLabel))
    for left, right in pairs:
        eqTab[left - 1, right - 1] = 1
        eqTab[right - 1, left - 1] = 1
    labelFlag = np.zeros(maxLabel)
    tempList = [1]
    eqList = []
    data_result = []
    for i in range(maxLabel):
        if labelFlag[i]:
            continue
        eqList = [i + 1]
        tempList = [i + 1]
        for k in range(maxLabel):
            if not tempList:
                break
            index = tempList.pop()
            for j in range(maxLabel):
                if eqTab[index - 1, j] and not labelFlag[j]:
                    tempList.append(j + 1)
                    eqList.append(j + 1)
            labelFlag[index - 1] = 1
        data_result.append(eqList)
    num_componets = len(data_result)
    if num_componets < 1:
        num_componets = max_labels
    else:
        connect_labels = sum([np.unique(np.array(data_result.__getitem__(i))).shape[0] for i in range(num_componets)])
        num_componets = num_componets + (max_labels - connect_labels)
    return num_componets


def cons_binary_convex(reward_maplist):
    reward_samples = []
    for reward_map in reward_maplist:
        batch_onesample = []
        for i in range(reward_map.shape[0]):
            num_componets = global_connect(reward_map[i])
            if num_componets == 1:
                cons_value = 1
            else:
                cons_value = 0
            batch_onesample.append(cons_value)
        batch_onesamplereward = np.array(batch_onesample)
        reward_samples.append(batch_onesamplereward)

    return reward_samples


def constraint_loss(x: torch.Tensor):
    lv = x[:, 1, :, :]
    bg = x[:, 0, :, :]
    sample_container = Bernoulli(lv)
    reward_maplist = []
    prob_list = []
    for num in range(5):
        reward_map = sample_container.sample()
        reward_maplist.append(reward_map)

        map_bg = torch.where(reward_map == 0, torch.Tensor([1]).to('cpu'), torch.Tensor([0]).to('cpu'))
        prob = map_bg * bg + reward_map * lv
        prob_list.append(prob)

    reward_list = cons_binary_convex(reward_maplist)

    assert len(reward_list) == len(prob_list)
    img_avgloss = []
    avg_reward = []
    for eachimg in range(prob_list[0].shape[0]):
        img_loss = []
        record_reward = 0
        for idx in range(len(reward_list)):
            record_reward = record_reward + reward_list[idx][eachimg]
            img_loss.append((-reward_list[idx][eachimg] * torch.log(prob_list[idx][eachimg] + 0.000001)).mean())
        img_avgloss.append(average_list(img_loss))
        avg_reward.append(record_reward/(len(reward_list)))
    global_reward = average_list(avg_reward)
    consfinal_loss = average_list(img_avgloss)

    return consfinal_loss, global_reward


input = torch.Tensor([[1,1,1,0,0,0,0,1],[0,0,0,0,1,1,0,0],[1,1,0,0,0,0,0,1]])
global_connect(input)