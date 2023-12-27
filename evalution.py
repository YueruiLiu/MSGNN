import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, label_ranking_loss

def accuracy_(label, output):
    zs = torch.sigmoid(output).to('cpu').data.numpy()
    ts = label.to('cpu').data.numpy()
    preds = list(map(lambda x: (x >= 0.5).astype(int), zs))

    preds_list, t_list = [], []
    preds_list = np.append(preds_list, preds)
    t_list = np.append(t_list, ts)

    acc = accuracy_score(t_list, preds_list)
    precision = precision_score(t_list, preds_list)
    recall = recall_score(t_list, preds_list)
    f1 = f1_score(t_list, preds_list)

    return acc, precision, recall, f1
