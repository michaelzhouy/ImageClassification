# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 8:44 上午
# @Author  : Michael Zhouy
import os
import time
import timm
from torch import nn
from torch import optim
from data_load import *
from util import accuracy, show_image
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'


path = '../data'
train_df = read_train(path)
test_df = read_test(path)

show_image(train_df.sample(5)['path'])

train_loader, val_loader = data_loader(train_df)
test_loader = test_loader(test_df)

devices = try_all_gpus()
model = timm.create_model(
    'efficientnet_b0',
    num_classes=137,
    pretrained=True,
    in_chans=3
)
model = nn.DataParallel(model, device_ids=devices).to(devices[0])
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader) / 10, gamma=0.95)

print('Epoch/Batch\t\tTrain: loss/Top1/Top3\t\tTest: loss/Top1/Top3')

for epoch in range(10):
    train_losss, train_acc1s, train_acc5s = [], [], []
    for i, data in enumerate(train_loader):
        scheduler.step()
        model.train()
        train_img, train_label = data
        optimizer.zero_grad()

        train_img = train_img.to(devices[0])
        train_label = train_label.view(-1).to(devices[0])

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

                    val_images = val_images.to(devices[0])
                    val_labels = val_labels.view(-1).to(devices[0])

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
            x_var = x.to(devices[0])
            y_var = y.to(devices[0])
            scores = model(x_var)
            pred.append(scores.data.cpu().numpy())
    pred = np.concatenate(pred, 0)
    print(tti)
    pred_tta.append(pred)

pred = np.mean(pred_tta, axis=0)


test_df['category_id'] = pred.argmax(1)
test_df['image_id'] = test_df['path'].apply(lambda x: x.split('/')[-1])

test_df[['image_id', 'category_id']].to_csv('submit_{}.csv'.format(time.strftime('%Y%m%d')), index=None)
