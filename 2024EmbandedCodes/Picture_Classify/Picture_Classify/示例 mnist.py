import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.modules.module import T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        # transforms.Resize((h,w)) Scale类似     缩放
        # transforms.CenterCrop((h,w))       以中心裁剪
        # transforms.RandomCrop((h,w))         随机裁剪
        # transforms.RandomHorizontalFlip()    随机反转
        # transforms.RandomVerticalFlip()
        # transforms.ToTensor()           转换成Tensor
        # transforms.ToPILImage()         转换成PIL
    ]
)

def initData(transform):
    data_raw_train = datasets.MNIST(root="data/", transform=transform, train=True)
    data_raw_test = datasets.MNIST(root="data/", transform=transform, train=False)

    data_loader_train = DataLoader(dataset=data_raw_train, shuffle=True, batch_size=64)
    data_loader_test = DataLoader(dataset=data_raw_test, shuffle=True, batch_size=64)

    return data_loader_train, data_loader_test

# 可视化
# images, labels = next(iter(data_loader_train))
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1,2,0)
# mean = std = [0.5, 0.5, 0.5]
# img = img * std + mean
#
# print("labels of img([:64]):", [labels[i] for i in range(64)])
# plt.imshow(img)


class FCNet(nn.Module):
    def __init__(self, input_size, hidden1_size=10, hidden2_size=10, output_size=10):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.ReLU(out)
        out = self.fc2(out)
        out = self.ReLU(out)
        out = self.fc3(out)
        return out


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
BATCH = 16
EPOCH = 10

pipeline = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

data_train, data_test = initData(pipeline)

INPUT_SIZE = 784
HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 256
model = FCNet(INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE).to(DEVICE)

lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

def train(train_loader):
    model.train()
    for batch_index, (data, label) in enumerate(train_loader):
        data, label = data.reshape(-1, 28*28).to(DEVICE), label.to(DEVICE)
        output = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, label)
        loss.backward()   # 梯度载给tensor
        optimizer.step()  # 根据梯度更新tensor

def test(test_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.reshape(-1, 28*28).to(DEVICE), label.to(DEVICE)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            _, predict = torch.max(output, dim=1)
            correct += predict.eq(label.view_as(predict)).sum().item()
        test_loss /= len(test_loader.dataset)


for epoch in range(1, EPOCH + 1):
    train(train_loader=data_train)
    test(test_loader=data_test)

