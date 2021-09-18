# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 8:44 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
import time
import cv2
from glob import glob
import timm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torch import nn
from torch import optim
from torch.utils.data.dataset import Dataset
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk返回一个元组, 元组的第一个元素为最大值, 第一个元素为最大值所在的索引
        # 这里应该是top3
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()  # 转置
        # expand_as()维度扩展为pred的维度
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(correct.shape)
        res = []
        for k in topk:
            # print(correct[:k].shape)
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))

        # print(res)
        return res


def read_train(path):
    train_df = pd.DataFrame({'path': glob(path + '/训练集/*/*')})
    train_df['label'] = train_df['path'].apply(lambda x: int(x.split('/')[-2]))
    return train_df


def read_test(path):
    test_df = pd.DataFrame({'path': glob(path + '/初赛_测试集/*')})
    test_df['label'] = 0
    return test_df


def show_image(paths):
    plt.figure(figsize=(10, 8))
    for idx, path in enumerate(paths):
        plt.subplot(1, len(paths), idx+1)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])


path = '../data'
train_df = read_train(path)
test_df = read_test(path)


show_image(train_df.sample(5)['path'])


class XunFeiDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(np.array([self.img_label[index]]))
        return img, label

    def __len__(self):
        return len(self.img_path)

train_df = train_df.sample(frac=1.0)

train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(
        train_df['path'].values[:-2000],
        train_df['label'].values[:-2000],
        transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(5, scale=[0.95, 1.05]),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ),
    batch_size=60,
    shuffle=True,
    num_workers=5
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(
        train_df['path'].values[-2000:],
        train_df['label'].values[-2000:],
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ),
    batch_size=60,
    shuffle=False,
    num_workers=5
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(
        test_df['path'].values[:],
        test_df['label'].values[:],
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    ),
    batch_size=60,
    shuffle=False,
    num_workers=5
)

model = timm.create_model(
    'efficientnet_b0',
    num_classes=137,
    pretrained=True,
    in_chans=3
)

model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) / 10, gamma=0.95)

print('Epoch/Batch\t\tTrain: loss/Top1/Top3\t\tTest: loss/Top1/Top3')

for epoch in range(10):
    train_losss, train_acc1s, train_acc5s = [], [], []
    for i, data in enumerate(train_loader):
        scheduler.step()
        model = model.train()
        train_img, train_label = data
        optimizer.zero_grad()

        train_img = train_img.cuda()
        train_label = train_label.view(-1).cuda()

        output = model(train_img)
        train_loss = loss_fn(output, train_label)

        train_loss.backward()
        optimizer.step()

        train_losss.append(train_loss.item())

        if i % int(100) == 0:
            val_losss, val_acc1s, val_acc5s = [], [], []

            with torch.no_grad():
                train_acc1, train_acc3 = accuracy(output, train_label, topk=(1, 3))
                train_acc1s.append(train_acc1.data.item())
                train_acc5s.append(train_acc3.item())

                for data in val_loader:
                    val_images, val_labels = data

                    val_images = val_images.cuda()
                    val_labels = val_labels.view(-1).cuda()

                    output = model(val_images)
                    val_loss = loss_fn(output, val_labels)
                    val_acc1, val_acc3 = accuracy(output, val_labels, topk=(1, 3))

                    val_losss.append(val_loss.item())
                    val_acc1s.append(val_acc1.item())
                    val_acc5s.append(val_acc3.item())

            logstr = '{0:2s}/{1:6s}\t\t{2:.4f}/{3:.4f}/{4:.4f}\t\t{5:.4f}/{6:.4f}/{7:.4f}'.format(
                str(epoch), str(i),
                np.mean(train_losss, 0), np.mean(train_acc1s, 0), np.mean(train_acc5s, 0),
                np.mean(val_losss, 0), np.mean(val_acc1s, 0), np.mean(val_acc5s, 0),
            )
            torch.save(model.state_dict(), '.model_{0}.pt'.format(epoch))
            print(logstr)


pred_tta = []
for tti in range(5):
    pred = []
    with torch.no_grad():
        for t, (x, y) in enumerate(test_loader):
            x_var = x.cuda()
            y_var = y.cuda()
            scores = model(x_var)
            pred.append(scores.data.cpu().numpy())
    pred = np.concatenate(pred, 0)
    print(tti)
    pred_tta.append(pred)

pred = np.mean(pred_tta, axis=0)


test_df['category_id'] = pred.argmax(1)
test_df['image_id'] = test_df['path'].apply(lambda x: x.split('/')[-1])

test_df[['image_id', 'category_id']].to_csv('submit_{}.csv'.format(time.strftime('%Y%m%d')), index=None)
