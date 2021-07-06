import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
import torchvision.models as models

import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    def __init__(self):
        self.reset()
 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# exit()
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 1

    table = []

    for j in range(7):
        
        # dataset_test = Mydataset('/home/nam/exp/image')

        # # split the dataset in train and test set
        # indices = torch.randperm(len(dataset)).tolist()
        # train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
        # test_dataset = torch.utils.data.Subset(dataset_test, indices[-50:])

        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                         batch_size=batch_size, 
        #                                         shuffle=True)

        # test_loader = torch.utils.data.DataLoader(dataset=dataset,
        #                                         batch_size=batch_size, 
        #                                         shuffle=False)

        model = models.resnext50_32x4d()
        model.fc = nn.Linear(in_features=2048, out_features=7, bias=True)
        model.load_state_dict(torch.load('/home/nam/workspace/WorkStageRecognition/model/model_state_dict_1', map_location=device))
        model.to(device)
        model.eval()

        # optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # criterion = nn.CrossEntropyLoss().to(device)


        train_loss_arr = []
        train_acc_arr = []

        val_loss_arr = []
        val_acc_arr = []

        val_loss_sum = 0
        val_acc_sum = 0

        losses = AverageMeter()
        top1 = AverageMeter()

        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        g = 0

        for i in range(100):
            # data = data.to(device)
            # target = target.to(device)
            # print(target)

            img = Image.open("/home/nam/workspace/WorkStageRecognition/images/test/{}/{}.png".format(j+1, i)).convert("RGB")
            img = torchvision.transforms.functional.to_tensor(img)
            img = img.unsqueeze(0)
            
            label = [i]
            label = torch.as_tensor(label, dtype=torch.int64)

            img = img.to(device)
            label = label.to(device)

            output = model(img) 
            _, predicted = torch.max(output, 1)

            if predicted[0] == 0:
                a  = a + 1
            elif predicted[0] == 1:
                b = b + 1
            elif predicted[0] == 2:
                c = c +1
            elif predicted[0] == 3:
                d = d + 1
            elif predicted[0] == 4:
                e = e + 1
            elif predicted[0] == 5:
                f = f + 1
            elif predicted[0] == 6:
                g = g + 1

        t = [a, b, c, d, e, f, g]
        table.append(t)
    print(table)
    fig, ax =plt.subplots(1,1)
    column_labels = ['start', 'cpu', 'cpu-lam', 'lam', 'lam-ssd', 'ssd', 'end']
    row_labels = ['start', 'cpu', 'cpu-lam', 'lam', 'lam-ssd', 'ssd', 'end']
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=table,rowLabels= row_labels,colLabels=column_labels,loc="center")

    plt.show()

if __name__ == "__main__":
    main()