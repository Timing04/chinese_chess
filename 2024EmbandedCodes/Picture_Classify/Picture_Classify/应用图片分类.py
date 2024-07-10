import os.path
import random

import torch
from torchvision.transforms import v2
import cv2
from LeCNN import LeCNN

device = torch.device("cuda:0")
cnn = torch.load("model/cat_dog_epoch5.pth").to(device)

trans = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.CenterCrop(224),
        v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
cnn.eval()

for _ in range(100):
    rand = random.randint(1, 12500)
    # rpic = cv2.imread(os.path.join("data/cat_dog/test/", str(rand)+".jpg"))
    rpic = cv2.imread(os.path.join("data/cat_dog/train/", "dog." + str(rand) + ".jpg"))
    pic = torch.unsqueeze(trans(rpic), dim=0).to(device)
    out = cnn(pic)

    if out[0][1] > out[0][0]:
        out = "dog"
        print("狗狗")
    else:
        out = "cat"
        print("猫猫")
    cv2.imshow(out, rpic)
    cv2.waitKey()
