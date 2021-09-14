import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil
import argparse
from dataset.load_data import get_image_df, my_collate_fn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataset.load_data import selfDataSet
from torchvision.models import resnet50, resnet101, vgg16
from models.multiscale_resnet import multiscale_resnet
from models.inception_resnet_v2 import pdr_inceptionresnetv2
from models.inception_v4 import inceptionv4
from models.xception import pdr_xception
from models.Ney import Net
from models.fpn import FPN101


def main():
    image_df = get_image_df(opt.img_root_train)
    train_df, valid_df = train_test_split(image_df, test_size=0.2, random_state=6, stratify=image_df['label'])

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(640, 640)),
            transforms.FixRandomRotate(bound='Random'),
            transforms.RandomHflip(),
            transforms.RandomVflip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=(640, 640)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    dataset = {
        'train': selfDataSet(
            img_path=train_df['image_name'],
            img_label=train_df['label'],
            transforms=data_transforms['train']
        ),
        'valid': selfDataSet(
            img_path=valid_df['image_name'],
            img_label=valid_df['label'],
            transforms=data_transforms['valid']
        )
    }

    dataloader = {
        'train': DataLoader(
            dataset['train'],
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            collate_fn=my_collate_fn
        ),
        'valid': DataLoader(
            dataset['valid'],
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_cpu,
            collate_fn=my_collate_fn
        )
    }


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    # 数据集路径
    parser.add_argument('--img_root_train', type=str, default="dataset/data/images/TRAIN/", help='whether to img root')
    # 模型及数据存储路径
    parser.add_argument('--checkpoint_dir', type=str, default='results/voc_resnet50/',
                        help='directory where model checkpoints are saved')
    # 网络选择
    parser.add_argument('--net', dest='net', type=str, default='resnet50', help='which net is chosen for training ')
    # 批次
    parser.add_argument('--batch_size', type=int, default=6, help='size of each image batch')
    # 学习率
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    # cuda设置
    parser.add_argument('--cuda', type=str, default="0", help='whether to use cuda if available')
    # CPU载入数据线程设置
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    # 暂停设置
    parser.add_argument('--resume', type=str, default=None, help='path to resume weights file')
    # 迭代次数
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    # 起始次数（针对resume设置）
    parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
    # 显示结果的间隔
    parser.add_argument('--print_interval', type=int, default=1, help='interval between print log')
    # 确认参数，并可以通过opt.xx的形式在程序中使用该参数
    opt = parser.parse_args()
    # 获取系统的cuda信息
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device

    # 网络模型的选择
    if opt.net == "multiscale":
        model = multiscale_resnet(6)
    elif opt.net == 'fpn':
        model = FPN101()
    elif opt.net == "resnet50":
        model = resnet50(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 20)
    elif opt.net == "resnet101":
        model = resnet101(pretrained=True)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        model.fc = torch.nn.Linear(model.fc.in_features, 6)
    elif opt.net == "xception":
        model = pdr_xception(6)
    elif opt.net == "inception_v4":
        model = inceptionv4(6)
    elif opt.net == "inception_resnet_v2":
        model = pdr_inceptionresnetv2(6)
    elif opt.net == "Net":
        model = Net(4)
    elif opt.net == "vgg16":
        model = vgg16(4)

    # 暂停选项
    if opt.resume:
        model.eval()
        print('resuming finetune from %s' % opt.resume)
        try:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(torch.load(opt.resume))
        except KeyError:
            model.load_state_dict(torch.load(opt.resume))
            model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Adam优化
    # optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=1e-5)
    # SGD优化
    optimizer = optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9, weight_decay=1e-4)
    # 损失函数
    criterion = CrossEntropyLoss()
    # 学习率衰减设置
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 最佳准确率置0
    best_precision = 0
    # 设置损失
    lowest_loss = 10000
    # 训练
    for epoch in range(opt.start_epoch, opt.epochs):
        # 训练
        acc_train, loss_train = train(
            dataloader['train'],
            model,
            criterion,
            optimizer,
            epoch,
            print_interval=opt.print_interval,
            filename=opt.checkpoint_dir
        )
        # 在日志文件中记录每个epoch的训练精度和损失
        with open(opt.checkpoint_dir + 'record.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, train_Precision: %.8f, train_Loss: %.8f\n' % (epoch, acc_train, loss_train))
        # 测试
        precision, avg_loss = validate(
            dataloader['val'],
            model,
            criterion,
            print_interval=opt.print_interval,
            filename=opt.checkpoint_dir
        )
        # 在日志文件中记录每个epoch的验证精度和损失
        with open(opt.checkpoint_dir + 'record_val.txt', 'a') as acc_file:
            acc_file.write('Epoch: %2d, Precision: %.8f, Loss: %.8f\n' % (epoch, precision, avg_loss))
            # 记录最高精度与最低loss
            is_best = precision > best_precision
            is_lowest_loss = avg_loss < lowest_loss
            best_precision = max(precision, best_precision)
            lowest_loss = min(avg_loss, lowest_loss)
            print('--' * 30)
            print(' * Accuray {acc:.3f}'.format(acc=precision), '(Previous Best Acc: %.3f)' % best_precision,
                  ' * Loss {loss:.3f}'.format(loss=avg_loss), 'Previous Lowest Loss: %.3f)' % lowest_loss)
            print('--' * 30)
            # 保存最新模型
            save_path = os.path.join(opt.checkpoint_dir, 'checkpoint.pth')
            torch.save(model.state_dict(), save_path)
            # 保存准确率最高的模型
            best_path = os.path.join(opt.checkpoint_dir, 'best_model.pth')
            if is_best:
                shutil.copyfile(save_path, best_path)
            # 保存损失最低的模型
            lowest_path = os.path.join(opt.checkpoint_dir, 'lowest_loss.pth')
            if is_lowest_loss:
                shutil.copyfile(save_path, lowest_path)