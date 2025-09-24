import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(128, 128),
    "hidden_layer_sizes":(32*32, 16*16),
    "activation_funcs":("relu", "relu"),
    "output_func":"sigmoid",
    "loss_func":"binary_cross_entropy",
    "batch_size":24,
    "learning_rate":0.005,
    "show_info":False,
    "epsilon":1e-6,
}

func = {
    "relu":F.relu,
    "sigmoid":F.sigmoid,
}

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for filename in os.listdir(root_dir):
            if filename.startswith('cat'):
                self.images.append(os.path.join(root_dir, filename))
                self.labels.append(0)  # 猫的标签为0
            elif filename.startswith('dog'):
                self.images.append(os.path.join(root_dir, filename))
                self.labels.append(1)  # 狗的标签为1

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        return self.transform(image) if self.transform else image, label

class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        w, h = config["regular_size"]
        last_dim = w * h
        self.layer = []
        for hidden_size in config["hidden_layer_sizes"]:
            fc = nn.Linear(last_dim, hidden_size)
            self.layer.append(fc)
            last_dim = hidden_size
        self.layer.append(nn.Linear(last_dim, 1))
        self.func_names = list(config["activation_funcs"])
        self.func_names.append(config["output_func"])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer, func_name in zip(self.layer, self.func_names):
            x = layer(x)
            x = func[func_name](x)
        return x

def load_data(path):
    w, h = config["regular_size"]
    bs = config["batch_size"]
    transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Grayscale(),
    ])
    ds = CustomDataset(path, transform)
    return DataLoader(ds, batch_size=bs)


if __name__ == "__main__":
    datas = load_data(config["train_path"])
    iterator = iter(datas)
    images, labels = next(iterator)

    net = BPNet()
    y = net.forward(images)
    print(y)
