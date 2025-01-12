import os
from PIL import Image
import numpy as np

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(512, 512),
    "hidden_layer_sizes":(512*512, 256*256, 128*128, 64*64, 1),
    "activation_funcs":("relu", "relu", "relu", "relu", "sigmoid"),
    "loss_func":"binary_cross_entropy",
    "batch_size":3,
}

activate_funcs = {
    "relu": lambda x: np.maximum(x, 0),
    "relu_back": lambda x: np.where(x > 0, 1, 0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "sigmoid_back": lambda  x: np.exp(-x) / (1 + np.exp(-x))**2,
}

loss_funcs = {
    "binary_cross_entropy": lambda y_true, y_pred: -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)),
    "binary_cross_entropy_back": lambda y_true, y_pred: (y_pred - y_true) / y_true.size,
}

def min_max_normalize(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

def regular_img(img):
    img_gray = img.convert('L')
    resized_img = img_gray.resize(config["regular_size"], Image.Resampling.BILINEAR)
    flatten_data = np.array(resized_img).flatten()
    regular_data = min_max_normalize(flatten_data)
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
    data = {"datas": imagines, "labels": labels, "data_num":len(imagines), "label_num":len(labels)}
    assert data["data_num"] == data["label_num"]
    return data

def init_parameters():
    parameters = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    for index in range(hidden_layers_num):
        if index == 0:
            (w, h) = config["regular_size"]
            parameters['b'+str(index)] = np.zeros(w * h)
        else:
            parameters['b'+str(index)] = np.zeros(np.array(config["hidden_layer_sizes"][index]))
            parameters['W'+str(index)] = np.random.randn(config["hidden_layer_sizes"][index],config["hidden_layer_sizes"][index-1]) * 0.01
    return parameters

# a = func(y), Y = W * X + b, X_(i) = a_(i-1)
def batch_forward(batch_data, parameters):
    cache = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    for index in range(hidden_layers_num):
        if index == 0:
            cache['y'+str(index)] = batch_data + parameters['b' + str(index)]
        else:
            cache['y'+str(index)] = np.dot(parameters['w'+str(index)],cache['a'+str(index-1)]) + parameters['b'+str(index)]
        activation_name = config['activation_funcs'][index]
        cache['a'+str(index)] = activate_funcs[activation_name](cache['y' + str(index)])
    return cache

# da / dW = da / dY * dY / dW = func_back(y) * W.T @ X
def batch_backward(batch_label, parameters, cache):
    grad_parameters = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    y_pred = cache['a'+str(len(config["hidden_layer_sizes"])-1)]
    loss_back = loss_funcs[config["loss_func"]+"_back"](batch_label, y_pred)
    for index in range(hidden_layers_num - 1, -1, -1):
        differ_func_name = config["activation_funcs"][index] + "_back"
        grad_parameters["db"+str(index)] = loss_back * activate_funcs[differ_func_name](cache['y' + str(index)])

def batch_update(data, parameters):
    batch_size = config["batch_size"]
    img_num = data["data_num"]
    for batch_index in range(0, img_num, batch_size):
        # 0, 3, 6
        batch_x = data["datas"][batch_index:min(img_num, batch_index+batch_size)]
        batch_y = data["labels"][batch_index:min(img_num, batch_index+batch_size)]



