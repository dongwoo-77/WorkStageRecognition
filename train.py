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



# trans = transforms.Compose([transforms.ToTensor])

# trainset = torchvision.datasets.ImageFolder(root="/home/nam/exp/image", transform = trans)
class Mydataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "1"))))
        # print(self.imgs)
    

    def __getitem__(self, idx):
        #print(img_path)
        img_path = os.path.join(self.root, "1", self.imgs[idx])
        # print(img_path.split('_')[-1].split('.')[0])
        img = Image.open(img_path).convert("RGB")

        label = [0, 1, 2, 3, 4]

        if int(img_path.split('_')[-1].split('.')[0]) < 755:
            label = 0
        if 754 < int(img_path.split('_')[-1].split('.')[0]) < 1480:
            label = 1
        if 1479 < int(img_path.split('_')[-1].split('.')[0]) < 2550:
            label = 2
        if 2549 < int(img_path.split('_')[-1].split('.')[0]) < 2970:
            label = 3
        if 2969 < int(img_path.split('_')[-1].split('.')[0]):
            label = 4

        label = torch.as_tensor(label, dtype=torch.int64)

        img = torchvision.transforms.functional.to_tensor(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

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
    batch_size = 8

    dataset = Mydataset('/home/nam/workspace/WorkStageRecognition/image')
    dataset_test = Mydataset('/home/nam/workspace/WorkStageRecognition/image')

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-50])
    test_dataset = torch.utils.data.Subset(dataset_test, indices[-50:])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

    model = models.resnext50_32x4d()

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss().to(device)


    train_loss_arr = []
    train_acc_arr = []

    val_loss_arr = []
    val_acc_arr = []


    for epoch in range(num_epochs):

        model.train()

        losses = AverageMeter()
        top1 = AverageMeter()

        for i, (data, target) in enumerate(train_loader):
            
            data = data.to(device)
            target = target.to(device)
            # print(data, target)

            output = model(data) 
            print(target)

            loss = criterion(output, target)

            output.float()
            loss.float()

            prec1 = accuracy(output.data, target)
            prec1 = prec1[0]

            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))

            if i % 100 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader), loss=losses))

            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        train_loss_arr.append(losses.avg)
        train_acc_arr.append(top1.avg)
        print("train result: Loss: {}, Acc: {}\n".format(losses.avg, top1.avg))


        model.eval()
        with torch.no_grad():
            val_loss_sum = 0
            val_acc_sum = 0

            losses = AverageMeter()
            top1 = AverageMeter()

            for i, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)

                output = model(data) 

                loss = criterion(output, target)

                output.float()
                loss.float()

                prec1 = accuracy(output.data, target)

                prec1 = prec1[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))

                if i % 100 == 0:
                    print('Test: [{0}/{1}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(test_loader), loss=losses, top1=top1))

            val_loss_arr.append(losses.avg)
            val_acc_arr.append(top1.avg)
            print("Validation result: Loss: {}, Acc: {}\n".format(losses.avg, top1.avg))

if __name__ == "__main__":
    main()