import os
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

data_dir = "./data/"

train_transforms = transforms.Compose([
    transforms.RandomRotation(35),  # 旋转图像
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 裁剪
    transforms.ColorJitter(),  # 修改亮度, 对比度和饱和度
    transforms.RandomHorizontalFlip(),  # 依概率水平翻转
    transforms.ToTensor(),  # 将image转换为tensor, 并归一化至[0, 1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(255), # 调整image的尺寸
    transforms.CenterCrop(224), # 中心裁剪
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pass transforms in data
train_data = datasets.ImageFolder(data_dir + '/training_set/training_set', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test_set/test_set', transform=test_transforms)

# Load train and test loaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=True)


def image_convert(img):
    img = img.clone().cpu().numpy()
    img = img.transpose(1, 2, 0)
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    img = img*std + mean
    return img


def plot_10():
    iter_ = iter(trainloader)
    images,labels = next(iter_)
    an_ = {'0':'cat','1':'dog'}

    plt.figure(figsize=(20, 10))
    for idx in range(10):
        plt.subplot(2, 5, idx+1)
        img = image_convert(images[idx])
        label = labels[idx]
        plt.imshow(img)
        plt.title(an_[str(label.numpy())])
        plt.show()

plot_10()

model = models.resnet50(pretrained=True)
print('model:\n', model)

fc_features = model.fc.in_features
print('Classifier input features: ', fc_features)

for param in model.parameters():
    param.requires_grad = False

# Define a new classifier to our cat and dog data
# Starts with 2048 because it's a requirement of ResNet
classifier = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

# Change model classifier
model.fc = classifier

# Criterion function
criterion = nn.NLLLoss()

# Set optimizer and learning rate
optimizer = optim.Adam(model.fc.parameters(), amsgrad=True)

# Send model to GPU
model.to('cuda')

# Train and validate our model

# Explain the model
print('Model: Resnet50 + 3 Layers + Dropout 0.2')
print('Criterion: NLLLoss / Optimizer: Adam')

# Set controls
epochs = 10
running_loss = 0
test_loss = 0
accuracy = 0

# Run epochs
for epoch in range(epochs):

    start_time = time.time()

    # Train loop    
    for images, labels in trainloader:
        # Move input and label tensors to GPU
        images, labels = images.to('cuda'), labels.to('cuda')

        # Zero gradients
        optimizer.zero_grad()

        # Get log probabilities
        logps = model(images)
        # Get loss from criterion
        loss = criterion(logps, labels)
        # Backward pass loss
        loss.backward()
        # Optimizer step
        optimizer.step()

        # Keeping track on loss
        running_loss += loss.item()

    # Put model in evaluation mode
    model.eval()

    # Stops gradient descent
    with torch.no_grad():
        # Run a test loop and get accuracy of our model
        for images, labels in testloader:
            # Move input and label tensors to GPU
            images, labels = images.to('cuda'), labels.to('cuda')

            # Get log probabilities
            logps = model(images)
            # Get loss from criterion
            loss = criterion(logps, labels)
            # Keeping track on testing loss
            test_loss += loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f'Epoch {epoch+1}/{epochs}.. '
          f'Train loss: {running_loss/len(trainloader):.3f}.. '
          f'Test loss: {test_loss/len(testloader):.3f}.. '
          f'Test accuracy: {accuracy/len(testloader):.3f}')

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # Reset controls
    running_loss = 0
    test_loss = 0
    accuracy = 0

    # Put model back in training mode
    model.train()


# Helper function to view predict images
def plot_predict_images():
    label_dict = ['cat', 'dog']

    iter_ = iter(testloader)
    images, labels = next(iter_)
    images = images.to('cuda')
    pred_labels = labels.to('cuda')

    img_out = model(images)
    value, index_val = torch.max(img_out, 1)

    fig = plt.figure(figsize=(35,9))
    for idx in np.arange(10):
        ax = fig.add_subplot(2,5,idx+1)
        plt.imshow(image_convert(images[idx]))
        label = labels[idx]
        pred_label = pred_labels[idx]
        ax.set_title('Act {},pred {}'.format(label_dict[label],label_dict[pred_label]))

plot_predict_images()
