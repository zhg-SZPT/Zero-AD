import torch
import numpy as np


def evaluation(preds_, targets_):
    n, n_class = preds_.shape
    Nc, Np, Ng = torch.zeros(n_class), torch.zeros(n_class), torch.zeros(n_class)
    for k in range(n_class):
        scores = preds_[:, k]
        targets = targets_[:, k]
        # targets[targets == -1] = 0
        Ng[k] = targets.sum()
        Np[k] = scores.sum()
        Nc[k] = (targets * scores).sum()
    Np[Np == 0] = 1
    OP = Nc.sum() / Np.sum()
    OR = Nc.sum() / Ng.sum()
    OF1 = (2 * OP * OR) / (OP + OR)

    CP = (Nc / Np).sum() / n_class
    CR = (Nc / Ng).sum() / n_class
    CF1 = (2 * CP * CR) / (CP + CR)
    return OP, OR, OF1, CP, CR, CF1


def calculate_mAP(preds, labels):
    no_examples = labels.shape[0]
    no_classes = labels.shape[1]

    ap_scores = np.empty((no_classes), dtype=np.float)
    for ind_class in range(no_classes):
        ground_truth = labels[:, ind_class]
        out = preds[:, ind_class]

        sorted_inds = np.argsort(out)[::-1] # in descending order
        tp = ground_truth[sorted_inds]
        fp = 1 - ground_truth[sorted_inds]
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        rec = tp / np.sum(ground_truth)
        prec = tp / (fp + tp)

        rec = np.insert(rec, 0, 0)
        rec = np.append(rec, 1)
        prec = np.insert(prec, 0, 0)
        prec = np.append(prec, 0)

        for ind in range(no_examples, -1, -1):
            prec[ind] = max(prec[ind], prec[ind + 1])

        inds = np.where(rec[1:] != rec[:-1])[0] + 1
        ap_scores[ind_class] = np.sum((rec[inds] - rec[inds - 1]) * prec[inds])

    return 100 * np.mean(ap_scores)