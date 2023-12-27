import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.dataloading import GraphDataLoader

import shutil

from config import *
from dataset.data_util import (
    get_data
)
from evalution import (
    accuracy_
)
from models.MSGNN import (
    Net
)
from models.utils import (
    Logger
)


def mean_std(name, list):
    final_mean, final_std = np.mean(list), np.std(list)
    logger.append(f" {name}: {final_mean:.4f}±{final_std:.4f}")
    return final_mean

# Check if GPU is available
cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

# Create a trainer class
class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self, dataset_train):
        loss_total = 0

        dataloader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        for batched_data, labels, h_CLR, maccs, morgan in dataloader:
            try:
                self.model.train()
                loss = self.model(batched_data, labels, h_CLR, maccs, morgan)
            except RuntimeError:
                pass
            else:
                self.optimizer.zero_grad()
                loss.mean().backward()
                if isinstance(self.model, nn.Module):
                    params = [p for p in self.model.parameters() if p.requires_grad]
                else:
                    params = self.model.params
                nn.utils.clip_grad_norm_(params, 1)
                self.optimizer.step()
                loss_total += loss.to('cpu').data.numpy()

        return loss_total


# Create a tester class
class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset_test):
        predict_list, true_list = [], []

        dataloader = GraphDataLoader(dataset_test)
        with torch.no_grad():
            for batched_data, labels, h_CLR, maccs, morgan in dataloader:
                out = self.model(batched_data, labels, h_CLR, maccs, morgan, train=False)
                predict_list.append(out)
                true_list.append(labels)
        predict_tensor = torch.stack(predict_list, 0).squeeze()
        true_tensor = torch.stack(true_list, 0).squeeze()

        auc, precision, recall, F1 = accuracy_(true_tensor, predict_tensor)

        return auc, precision, recall, F1


FOLD = 10
args = make_args()

Accuracy_Avg, Precision_Avg, Recall_Avg, F1_Avg = [], [], [], []
for fold in range(FOLD):
    fingerprint_dict, dataset_train, dataset_test = get_data(fold+1)

    print('------------------------------------------------------------------')
    print(f'Fold-{fold+1}:')

    n_fingerprint = len(fingerprint_dict)

    # Create and train model
    model = Net(n_fingerprint, args, device).to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    start = timeit.default_timer()

    logger = Logger('./logs/MSGNN_'+str(fold+1)+'.log')
    logger.append(vars(args))
    logger.append('Epoch \t Time(sec) \t Loss_train \t ACC_test \t Precision_test \t Recall_test \t F1_test')

    Accuracy_list, Precision_list, Recall_list, F1_list = [], [], [], []
    max_accuracy_index = 0
    for epoch in range(args.epochs):
        if (epoch + 1) % args.decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= args.lr_decay

        loss = trainer.train(dataset_train)
        acc_test, precision, recall, f1_test = tester.test(dataset_test)

        Accuracy_list.append(acc_test)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1_test)

        if acc_test > Accuracy_list[max_accuracy_index]:
            max_accuracy_index = epoch
            if (args.save_model):
                torch.save(model.state_dict(), 'results/every_fold_model/MSGNN'+str(fold+1)+'.pt')

        end = timeit.default_timer()
        time = end - start

        logger.append('%d \t\t %.4f \t %.4f \t\t %.4f \t %.4f \t\t\t %.4f \t\t %.4f' % (epoch, time, loss,
                                                                                acc_test, precision,
                                                                                recall, f1_test,
                                                                                ))

    accuracy_avg = mean_std('Accuracy', Accuracy_list)
    precision_avg = mean_std('Precision', Precision_list)
    recall_avg = mean_std('Recall', Recall_list)
    f1_avg = mean_std('F1', F1_list)

    Accuracy_Avg.append(accuracy_avg)
    Precision_Avg.append(precision_avg)
    Recall_Avg.append(recall_avg)
    F1_Avg.append(f1_avg)


print('==========================================================')
print('The final performances of MSGNN are:')
print('Accuracy', np.mean(Accuracy_Avg))
print('Precision', np.mean(Precision_Avg))
print('Recall', np.mean(Recall_Avg))
print('F1', np.mean(F1_Avg))

if (args.save_model):
    max_accuracy_index = Accuracy_Avg.index(max(Accuracy_Avg))
    src = f'results/every_fold_model/MSGNN{max_accuracy_index}.pt'
    dst = 'results/best_model/MSGNN.pt'
    shutil.copy(src, dst)