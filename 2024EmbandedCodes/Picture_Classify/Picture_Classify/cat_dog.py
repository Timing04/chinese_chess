import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import os
import cv2

path = "data/train/train"
transform = v2.Compose([v2.ToTensor(),
                        v2.CenterCrop(224),
                        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# 数据加载，模型训练等代码需要自己补全

model = models.vgg16(pretrained=True)

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

for parma in model.parameters():
    parma.requires_grad = False

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

use_gpu = True
if use_gpu:
    model = model.cuda()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())


# import torchvision
# dataset = torchvision.

class Pets(Dataset):
    def __init__(self):
        self.picDatas = []
        self.picDatas_Class = []
        picFiles = os.listdir(path)
        # print(picFiles)
        for i in picFiles[0:60]:
            rowPic = cv2.imread(path + "/" + i)
            picData = transform(rowPic)
            self.picDatas.append(picData)
            if i[0:2] == "cat":
                self.picDatas_Class.append("cat")
            elif i[0:2] == "dog":
                self.picDatas_Class.append("dog")
            print(i, "已处理")
        self.ts_picDatas = torch.tensor(self.picDatas)
        self.ts_picDatas_Class = torch.tensor(self.picDatas_Class)
        print("[Success] 数据集初始化成功")

    def __getitem__(self, item_index):
        return self.ts_picDatas[item_index], self.ts_picDatas_Class[item_index]

    def __len__(self):
        return len(self.ts_picDatas)


pets = Pets()
print("[info] 数据集取样：", pets.ts_picDatas[1])
print("[info] 数据集长度：", len(pets))

pets_train, pets_test = random_split(dataset=pets, lengths=[0.85,0.15])
train_dataloader = DataLoader(pets_train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(pets_test, batch_size=128, shuffle=True)

model.train()

