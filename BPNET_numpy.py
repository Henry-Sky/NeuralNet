import os
from PIL import Image
import numpy as np

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(512, 512),
    "hidden_layer_sizes":(512*512, 256*256, 128*128, 64*64),
    "activation_funcs":("relu", "relu", "relu", "sigmoid"),
}

activations = {
    "relu": lambda x: np.maximum(x, 0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
}

def min_max_normalize(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(x - min_value) / (max_value - min_value) for x in data]
    return normalized_data

def regular_img(img):
    img_gray = img.convert('L')
    resized_img = img_gray.resize(config["regular_size"], Image.Resampling.BILINEAR)
    flatten_data = np.array(resized_img).flatten()
    assert len(flatten_data) == config["regular_size"][0]*config["regular_size"][1]
    regular_data = min_max_normalize(flatten_data)
    assert len(regular_data) == config["regular_size"][0] * config["regular_size"][1]
    return regular_data

def load_data(path):
    imagines = []
    labels = []
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))
        regular_data = regular_img(img)
        imagines.append(regular_data)
        if filename[:3].lower() == "dog":
            labels.append(1)
        elif filename[:3].lower() == "cat":
            labels.append(0)
        else:
            raise ValueError(f"Unexpected file name: {filename}")
    data = {"datas": imagines, "labels": labels}
    return data

def init_parameters():
    parameters = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    for index in range(hidden_layers_num):
        if index == 0:
            parameters['b'+str(index)] = np.zeros((config["regular_size"][0] * config["regular_size"][1]))
        else:
            parameters['b'+str(index)] = np.zeros(np.array(config["hidden_layer_sizes"][index]))
            parameters['w'+str(index)] = np.random.randn(config["hidden_layer_sizes"][index],config["hidden_layer_sizes"][index-1]) * 0.01
    return parameters

def forward(train_datas,parameters):
    cache = {}
    for i in range(len(config["hidden_layer_sizes"])):
        if i == 0:
            cache['y'+str(i)] = train_datas + parameters['b'+str(i)]
        else:
            cache['y'+str(i)] = np.dot(parameters['w'+str(i)],cache['a'+str(i-1)]) + parameters['b'+str(i)]
        activation_name = config['activation_funcs'][i]
        cache['a'+str(i)] = activations[activation_name](cache['y'+str(i)])
    return cache