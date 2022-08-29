import numpy as np
import torch


def del_tensor_ele_n(arr, index, n):
    """
    arr: 输入tensor
    index: 需要删除位置的索引
    n: 从index开始，需要删除的列数
    """
    arr1 = arr[:, 0:index]
    arr2 = arr[:, index+n:]
    return torch.cat((arr1, arr2), dim=1)


def avoid_perfect_And_defect(scores):
    new_scores = scores.clone()
    for i, row in enumerate(scores):
        if row[0] > 0:
            for j in range(1, len(row)):
                if row[j] > 0:
                    index = torch.argmax(row)
                    if index == 0:
                        new_scores[i, 1:] = 0
                    else:
                        new_scores[i, 0] = 0
                    break
    return new_scores