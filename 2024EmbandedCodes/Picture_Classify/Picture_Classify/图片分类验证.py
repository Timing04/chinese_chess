import os

import cv2
import torch
from torchvision.transforms import v2

device = torch.device("cuda:0")
cnn = torch.load("model/cat_dog_epoch5.pth").to(device)

trans = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(224),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

picfiles = os.listdir("data/cat_dog/train")
allcrt = 0
for batch in range(round(len(picfiles)/128)):
    out = [cnn(torch.unsqueeze(trans(cv2.imread(os.path.join("data/cat_dog/train", pic))).to(device),dim=0)) for pic in picfiles[batch*128:batch*128+128]]
    # pics = []
    # for pic in picfiles:
    #     pics.append(trans(cv2.imread(os.path.join("data/cat_dog/train", pic))))

    # print("start clssifing")
    # out = cnn(pics)
    # for i in out:
    #     print(i)
    clss = ["cat" if i[0][0] > i[0][1] else "dog" for i in out]
    crt = 0
    for i, cls in enumerate(clss):
        if cls == picfiles[batch*128+i][0:3]:
            allcrt += 1
            crt += 1
        else:
            print("Pic {} is {}, recognized as {}".format(picfiles[batch*128+i], picfiles[batch*128+i][0:3], cls))

    print("{} correct in {} pics, acc is {}".format(crt, len(clss), crt/len(clss)))

print(allcrt)