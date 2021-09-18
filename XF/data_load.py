# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 9:10 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
from glob import glob
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image


def my_collate_fn(batch):
    """
    batch中每个元素形如(data, label)
    :param batch:
    :return:
    """
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)  # 用默认方式拼接过滤后的batch数据


def read_train(path):
    train_df = pd.DataFrame({'path': glob(path + '/训练集/*/*')})
    train_df['label'] = train_df['path'].apply(lambda x: int(x.split('/')[-2]))
    return train_df


def read_test(path):
    test_df = pd.DataFrame({'path': glob(path + '/初赛_测试集/*')})
    test_df['label'] = 0
    return test_df


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


def data_loader(train_df):
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
        num_workers=5,
        collate_fn=my_collate_fn
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
        num_workers=5,
        collate_fn=my_collate_fn
    )
    return train_loader, val_loader


def test_loader(test_df):
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
        num_workers=5,
        collate_fn=my_collate_fn
    )
    return test_loader
