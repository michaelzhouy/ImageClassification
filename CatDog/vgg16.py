import os
import time
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'


data_dir = "./data"
data_str = ["test_set", "training_set"]
data_transform = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])
img_datasets = {
    x: datasets.ImageFolder(
        root=data_dir + '/{}/{}'.format(x, x),
        transform=data_transform)
    for x in data_str
}
dataloader = {
    x: torch.utils.data.DataLoader(
        dataset=img_datasets[x],
        batch_size=16,
        shuffle=True
    )
    for x in data_str
}

X_example, y_example = next(iter(dataloader["training_set"]))

print("X_example's number:{}".format(len(X_example)))
print("y_example's number:{}".format(len(y_example)))
index_classes = img_datasets["training_set"].class_to_idx
print(index_classes)
example_classes = img_datasets["training_set"].classes
print(example_classes)


img = torchvision.utils.make_grid(X_example)
img = img.numpy().transpose([1, 2, 0])
print([example_classes[i] for i in y_example])
plt.imshow(img)
plt.show()  # visualization


model = models.vgg16(pretrained=True)
print('model: \n', model)


for parma in model.parameters():
    parma.requires_grad = False

fc_features = model.classifier[0].in_features
print('classifier input features: ', fc_features)

model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=25088, out_features=1024 * 4, bias=True),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5, inplace=False),
    torch.nn.Linear(in_features=1024 * 4, out_features=1024 * 4, bias=True),
    torch.nn.ReLU(inplace=True),
    torch.nn.Dropout(p=0.5, inplace=False),
    torch.nn.Linear(in_features=1024 * 4, out_features=2, bias=True),
)
model = model.cuda()
lf = torch.nn.CrossEntropyLoss()
op = torch.optim.Adam(model.classifier.parameters(), lr=1e-5)
print(model)


epoch_n = 5
time_open = time.time()

for epoch in range(epoch_n):
    print('Epoch {}/{}'.format(epoch + 1, epoch_n))
    print('-----------')

    for phase in data_str:
        if phase == 'training_set':
            print('Training...')
            model.train(True)
        else:
            print('Validing...')
            model.train(False)
        running_loss = 0.0
        running_correct = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            X, y = X.cuda(), y.cuda()
            y_pred = model(X)

            _, pred = torch.max(y_pred.data, 1)

            op.zero_grad()
            loss = lf(y_pred, y)

            if phase == 'training_set':
                loss.backward()
                op.step()

            running_correct += torch.sum(pred == y.data)
            running_loss += loss.data
            if batch % 500 == 0 and phase == 'training_set':
                print('Batch {},Train Loss:{:.4f},Train Acc:{:.4f}%'.format(
                    batch, running_loss / batch, 100 * running_correct / batch / 16))

        epoch_loss = running_loss * 16 / len(img_datasets[phase])
        epoch_acc = running_correct * 100 / len(img_datasets[phase])

        print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))

time_end = time.time() - time_open
print(time_end)

torch.save(model.state_dict(), 'model_params.pth')

print("Valid...")
model.train(False)
running_loss = 0.0
running_correct = 0

for batch, data in enumerate(dataloader["test_set"], 1):
    X, y = data
    X, y = Variable(X.cuda()), Variable(y.cuda())
    y_pred = model(X)

    _, pred = torch.max(y_pred.data, 1)

    op.zero_grad()
    loss = lf(y_pred, y)

    running_correct += torch.sum(pred == y.data)
    running_loss += loss.data

epoch_loss = running_loss * 16 / len(img_datasets["test_set"])
epoch_acc = running_correct * 100 / len(img_datasets["test_set"])

print("{} Loss:{:.4f} Acc:{:.4f}%".format("valid", epoch_loss, epoch_acc))
