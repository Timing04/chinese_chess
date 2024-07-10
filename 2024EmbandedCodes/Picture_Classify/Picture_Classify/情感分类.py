import torch
from torchvision.transforms import v2
import torch.utils.data as td

class Datas(td.Dataset):
    def __init__(self):
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),

        ])
