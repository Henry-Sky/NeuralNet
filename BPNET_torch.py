import os
import torch
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.optim as optim

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 优先选择 CUDA 设备
        print("Using CUDA device.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # 如果 CUDA 不可用，选择 MPS 设备
        print("Using MPS device.")
    else:
        device = torch.device("cpu")  # 如果 CUDA 和 MPS 都不可用，选择 CPU
        print("Using CPU device.")
    return device

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
    "device":get_device(),
}

activate_funcs = {
    "relu":F.relu,
    "sigmoid":F.sigmoid,
}

loss_funcs = {
    "binary_cross_entropy": lambda y_true, y_out: - (y_true * torch.log(y_out) + (1 - y_true) * torch.log(1 - y_out)),
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
        self.layers = nn.ModuleList()
        for hidden_size in config["hidden_layer_sizes"]:
            fc = nn.Linear(last_dim, hidden_size).to(config["device"])
            self.layers.append(fc)
            last_dim = hidden_size
        self.layers.append(nn.Linear(last_dim, 1).to(config["device"]))
        self.func_names = list(config["activation_funcs"])
        self.func_names.append(config["output_func"])

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for layer, func_name in zip(self.layers, self.func_names):
            x = layer(x)
            x = activate_funcs[func_name](x)
        return torch.flatten(x)

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

def count_loss(y_true, y_out):
    assert (y_true.shape == y_out.shape)
    func = config["loss_func"]
    losses = loss_funcs[func](y_true, y_out)
    return torch.mean(losses)

if __name__ == "__main__":
    net = BPNet()
    dataloader = load_data(config["train_path"])
    optimizer = optim.SGD(net.parameters(), lr=config["learning_rate"])

    for iter in tqdm(range(100)):
        loss_list = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(config["device"]), labels.to(config["device"])
            output = net.forward(inputs)
            loss = count_loss(labels, output)
            loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = sum(loss_list) / len(loss_list)
        print(f"iter: {iter+1}, avg_loss: {avg_loss:.8f}")