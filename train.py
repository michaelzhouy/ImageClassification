# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 11:27 上午
# @Author  : Michael Zhouy
import numpy as np
import pandas as pd
import random
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import timm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, lr_scheduler
from torchvision import transforms
from PIL import ImageFile, Image
from glob import glob
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True


label_id_dic = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5
}


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self, lb_pos=0.9, lb_neg=0.005, reduction='mean', lb_ignore=255):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1 - lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs * label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs * label, dim=1)
        return loss


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(net, num_classes=6):
    num_classes = num_classes
    if net == 'swin_base_patch4_window7_224':
        net = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif net == 'swin_large_patch4_window7_224':
        net = timm.create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif net == 'swin_base_patch4_window12_384':
        net = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=num_classes)
    elif net == 'swin_large_patch4_window12_384':
        net = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=num_classes)
    return net


def eval_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def load_data(data_dir):
    train = pd.read_csv(data_dir + 'train.csv')
    train['file_name'] = train['file_name'].map(lambda x: x.split('.')[0])
    train['label'] = train['category'].map(label_id_dic)
    train_img = pd.DataFrame({'path': glob(data_dir + 'train/*.*')})
    train_img['file_name'] = train_img['path'].map(lambda x: x.split('/')[-1].split('.')[0])
    train = train.merge(train_img, on='file_name', how='left')

    test = pd.DataFrame({'path': glob(data_dir + 'validation/*.*')})
    test['id'] = test['path'].map(lambda x: x.split('/')[-1])
    test['label'] = -1
    print('train Null counts: ', train.isnull().sum())
    print('train shape: ', train.shape)
    print('test  shape: ', test.shape)
    return train, test


def split_data(data, test_size=0.25):
    train_df, valid_df = train_test_split(data, test_size=test_size, random_state=6, stratify=data['label'])
    return train_df, valid_df


class selfDataset(Dataset):
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


def my_collate_fn(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def data_loader(train_df, valid_df, test_df, batch_size=56, input_size=224, num_workers=16):
    train_loader = torch.utils.data.DataLoader(
        selfDataset(
            train_df['path'].values,
            train_df['label'].values,
            transforms.Compose([
                transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    # transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    # transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
            ])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )

    valid_loader = torch.utils.data.DataLoader(
        selfDataset(
            valid_df['path'].values,
            valid_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        selfDataset(
            test_df['path'].values,
            test_df['label'].values,
            transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        # collate_fn=my_collate_fn
    )
    return train_loader, valid_loader, test_loader


def run(data_dir, model_dir, EPOCH, model, batch_size, LR, num_workers, input_size):
    train, test = load_data(data_dir=data_dir)
    train_df, valid_df = split_data(train, test_size=2000)

    train_loader, val_loader, test_loader = data_loader(train_df, valid_df, test, batch_size=batch_size,
                                                        input_size=input_size, num_workers=num_workers)

    print('model: ', model)
    net = load_model(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 训练使用多GPU，测试单GPU
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    net = net.to(device)

    loss_fn = CrossEntropyLoss()
    # optimizer = Adam(net.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-9)
    optimizer = SGD((net.parameters()), lr=LR, momentum=0.9, weight_decay=0.0004)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)
    print('Start training...')
    best_acc = 0
    for epoch in range(EPOCH):
        print('\nEPOCH: %d' % (epoch + 1))
        sum_loss, correct, total = 0.0, 0.0, 0.0
        true, pred = [], []
        net.train()
        for i, data in enumerate(train_loader):
            train_img, train_label = data

            train_img = train_img.to('cuda')
            train_label = torch.tensor(train_label, dtype=torch.long)
            train_label = train_label.view(-1).to('cuda')

            optimizer.zero_grad()
            output = net(train_img)
            # print('shape: ', train_label.shape, output.shape)
            train_loss = loss_fn(output, train_label)

            train_loss.backward()
            optimizer.step()

            sum_loss += train_loss.item()
            _, predicted = torch.max(output.data, 1)
            total += train_label.size(0)
            correct += predicted.eq(train_label.data).cpu().sum()
            true += train_label.tolist()
            pred += predicted.tolist()
        train_acc = float(correct) / float(total)
        train_metric = eval_metric(true, pred)
        print('[Epoch: %d, iter: %d] Loss: %.03f | Acc: %.5f | Metric: %.5f'
              % (epoch + 1, (i + 1), sum_loss / (i + 1), train_acc, train_metric))

        valid_sum_loss, correct, total = 0.0, 0.0, 0.0
        true, pred = [], []
        net.eval()
        for i, data in enumerate(val_loader):
            val_images, val_labels = data
            val_images = val_images.to('cuda')
            val_labels = val_labels.view(-1).to('cuda')
            with torch.no_grad():
                output = net(val_images)
                val_loss = loss_fn(output, val_labels)

                valid_sum_loss += val_loss.item()
                _, predicted = torch.max(output.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).cpu().sum()
                true += val_labels.tolist()
                pred += predicted.tolist()
        valid_acc = float(correct) / float(total)
        valid_metric = eval_metric(true, pred)
        print('Valid Loss: %.03f | Valid Acc: %.5f | Metric: %.5f' %
              (valid_sum_loss / (i + 1), valid_acc, valid_metric))

        # scheduler.step(acc)
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('Saving model...')
            print('EPOCH = %03d, Accuracy = %.5f' % (epoch + 1, valid_acc))
            if not os.path.exists(os.path.join(model_dir, model)):
                os.makedirs(os.path.join(model_dir, model))
            torch.save(net.module, '%s/model.pth' % (os.path.join(model_dir, model)))
            save_info = {
                'optimizer': optimizer.state_dict(),
                'model': net.module.state_dict()
            }
            torch.save(save_info, '%s/params.pkl' % (os.path.join(model_dir, model)))

    print("Training Finished, TotalEPOCH=%d" % EPOCH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据路径
    parser.add_argument('--data_dir', type=str, default='../data/', help='whether to img root')
    # 模型保存路径
    parser.add_argument('--model_dir', type=str, default='./model/model_pth/', help='whether to img root')
    # 迭代次数
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    # 模型
    parser.add_argument('--model', dest='model', type=str, default='resnet50', help='which net is chosen for training ')
    # 批次
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    # 图片大小
    parser.add_argument('--input_size', type=int, default=224, help='shape of each image')
    # 学习率
    parser.add_argument('--LR', type=float, default=0.01, help='LR')
    # CPU载入数据线程设置
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of cpu threads to use during batch generation')
    # cuda设置
    parser.add_argument('--cuda', type=str, default='0,1', help='whether to use cuda if available')
    # 确认参数，并可以通过opt.xx的形式在程序中使用该参数
    opt = parser.parse_args()
    # 获取系统的cuda信息
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    print('cuda: ', opt.cuda)
    print('epochs: ', opt.epochs)
    print('batch_size: ', opt.batch_size)
    print('LR: ', opt.LR)
    set_seed(10)

    run(data_dir=opt.data_dir, model_dir=opt.model_dir, EPOCH=opt.epochs, model=opt.model,
        batch_size=opt.batch_size, LR=opt.LR, num_workers=opt.num_workers, input_size=opt.input_size)
