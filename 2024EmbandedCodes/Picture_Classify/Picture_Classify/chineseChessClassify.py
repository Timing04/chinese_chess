import os
import pickle

import cv2
import torch
from torch import nn, optim
from torch.utils import data as td
import torch.functional as F

from torchvision.transforms import v2

DEVICE = torch.device("cuda")

class Chess(td.Dataset):
    def __init__(self):
        self.trans = v2.Compose([
            # 转换为Tensor
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            # 中心裁切
            v2.CenterCrop(224),
            # 归一化
            v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.pics = []
        self.picClasses = []

        picDirpath = "D:\Projects\Picture_Classify\data"
        picFileNames = os.listdir(picDirpath)

        for picFileName in picFileNames:

            if os.path.exists(os.path.join("cache", picFileName)):
                with open(os.path.join("cache", picFileName), mode="rb") as picDataFile:
                    transpic = pickle.load(picDataFile)
            else:
                print("not find" + os.path.join("cache", picFileName))
                picFilePath = os.path.join(picDirpath, picFileName)
                rowpic = cv2.imread(picFilePath)
                transpic = self.trans(rowpic)
                with open(os.path.join("cache", picFileName), mode="wb") as picDataFile:
                    pickle.dump(transpic, picDataFile)

            #象棋分类
            self.pics.append(transpic)
            if picFileName[0:1] == "1" :
                self.picClasses.append(1)
            elif picFileName[0:1] == "2" :
                self.picClasses.append(2)
            elif picFileName[0:1] == "3" :
                self.picClasses.append(3)
            elif picFileName[0:1] == "4" :
                self.picClasses.append(4)
            elif picFileName[0:1] == "5" :
                self.picClasses.append(5)
            elif picFileName[0:1] == "6" :
                self.picClasses.append(6)
            elif picFileName[0:1] == "7" :
                self.picClasses.append(7)
            elif picFileName[0:1] == "8" :
                self.picClasses.append(8)
            elif picFileName[0:1] == "9" :
                self.picClasses.append(9)
            elif picFileName[0:1] == "10" :
                self.picClasses.append(20)

        self.tPics = torch.stack(self.pics)
        self.tPicClasses = torch.tensor(self.picClasses)
        print("数据集大小", self.tPics.size())
        print("数据集标签大小", self.tPicClasses.size())

    def __getitem__(self, item):
        return self.tPics[item], self.tPicClasses[item]

    def __len__(self):
        return len(self.tPics)

#  实例化数据集，划分并转换为Loader
BATCHSIZE = 160
chsee = Chess()
chess_train, chess_test = td.random_split(chsee, lengths=[0.9, 0.1])
trainloader = td.DataLoader(chess_train, batch_size=BATCHSIZE, shuffle=True)
testloader = td.DataLoader(chess_test, batch_size=BATCHSIZE, shuffle=True)
print("数据集准备完成\n\n")


# 然后设计模型
from LeCNN import LeCNN

#  实例化CNN
cnn = LeCNN().to(DEVICE)
print(cnn)
print("模型准备完成\n\n")

# 最后开始训练
#  设置训练损失与优化器
opti = optim.Adam(cnn.parameters())
loss = nn.CrossEntropyLoss()

EPOCH = 80
cnn.train()
for epoch in range(EPOCH):
    lsall = 0
    lscnt = 0
    for step, (batch_x, batch_label) in enumerate(trainloader):
        batch_x, batch_label = batch_x.to(DEVICE), batch_label.to(DEVICE)
        out = cnn(batch_x)
        ls = loss(out, batch_label)
        opti.zero_grad()
        ls.backward()
        opti.step()
        lsall += ls
        lscnt += 1
        # 较少代码写法
        # opti.zero_grad()
        # loss(cnn(batch_x), batch_label).backward()
        # opti.step()
    print("epoch: {}\tloss: {}".format(epoch, lsall/lscnt))
    torch.save(cnn, "model/cat_dog_epoch"+str(epoch)+".pth")

models = os.listdir("model")
for model in models:
    cnn = torch.load("model/"+model)

    cnn.eval()
    ls_test = 0
    p = 0
    for step_test, (batch_test_x, batch_test_label) in enumerate(testloader):
        p+=1
        batch_test_x, batch_test_label = batch_test_x.to(DEVICE), batch_test_label.to(DEVICE)
        out_test = cnn(batch_test_x)
        # print(out_test)
        ls_test += loss(out_test, batch_test_label)
        _, predict = torch.max(out_test, dim=1)
        # print(predict, batch_test_label)

    print(ls_test/p, model)
