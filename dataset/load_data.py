# -*- coding: utf-8 -*-
# @Time    : 2021/9/14 10:51 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
import torch
from glob import glob
from PIL import Image
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset


class selfDataSet(Dataset):
    def __init__(self, img_path, img_label, transforms=None):
        self.img_path = img_path
        self.img_label = img_label
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transforms is not None:
            img = self.transforms(img)

        label = self.img_label[index]
        return img, label

    def __len__(self):
        return len(self.img_path)


def get_image_df(image_path):
    images = glob(image_path + '/*/*.jpg')
    df_image = pd.DataFrame(images, columns=['image_name'])
    df_image['label'] = df_image['image_name'].map(lambda x: x.split('/')[-2])
    print(df_image['label'].value_counts())
    return df_image


def my_collate_fn(batch):
    # 过滤为None的数据
    batch = list(filter(lambda x: x[0] is not None, batch))
    if len(batch) == 0:
        return torch.Tensor()
    return default_collate(batch)
