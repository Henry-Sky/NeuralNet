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
    "learning_rate":0.001,
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
    (w, h) = config["regular_size"]
    for index in range(hidden_layers_num):
        if index == 0:
            parameters['b'+str(index)] = np.zeros(w * h)
        else:
            parameters['b'+str(index)] = np.zeros(np.array(config["hidden_layer_sizes"][index]))
            parameters['W'+str(index)] = np.random.randn(config["hidden_layer_sizes"][index],config["hidden_layer_sizes"][index-1]) * 0.01
    return parameters

def batch_forward(batch_data, parameters):
    cache = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    batch_size = batch_data.shape[0]
    trans_mat = np.ones(batch_size, 1).T
    for index in range(hidden_layers_num):
        if index == 0:
            cache['Y'+str(index)] = batch_data + trans_mat @ parameters['b' + str(index)]
        else:
            cache['Y'+str(index)] = cache['A'+str(index-1)] @ parameters['W'+str(index)] + trans_mat @ parameters['b'+str(index)]
        activation_name = config['activation_funcs'][index]
        cache['A'+str(index)] = activate_funcs[activation_name](cache['Y' + str(index)])
    return cache

def batch_backward(batch_label, parameters, cache):
    grad_parameters = {}
    hidden_layers_num = len(config["hidden_layer_sizes"])
    batch_label_pred = cache['a'+str(len(config["hidden_layer_sizes"])-1)]
    loss = np.mean(loss_funcs[config["loss_func"]](batch_label_pred, batch_label))
    loss_back = np.mean(loss_funcs[config["loss_func"]+"_back"](batch_label_pred, batch_label))

    for index in range(hidden_layers_num - 1, -1, -1):
        differ_func =activate_funcs[config["activation_funcs"][index] + "_back"]
        if index == hidden_layers_num - 1:
            grad_parameters["db"+str(index)] = loss_back * differ_func(cache['Y'+str(index)])
        else:
            grad_parameters["db"+str(index)] = loss_back * differ_func(cache['Y'+str(index)]) *\
                                               np.outer(parameters['W'+str(index+1)], grad_parameters["db"+str(index+1)])
        if index == 0:
            continue
        else:
            grad_parameters["dW"+str(index)] = loss_back * np.outer(grad_parameters["db"+str(index)], cache['A'+str(index-1)])
    return grad_parameters


def batch_update(data, parameters):
    batch_size = config["batch_size"]
    img_num = data["data_num"]
    for batch_index in range(0, img_num, batch_size):
        # 0, 3, 6
        batch_x = data["datas"][batch_index:min(img_num, batch_index+batch_size)]
        cache = batch_forward(batch_x, parameters)
        batch_y = data["labels"][batch_index:min(img_num, batch_index+batch_size)]
        grad_parameters = batch_backward(batch_y, parameters, cache)
        for key in parameters.keys():
            parameters[key] -= config["learning_rate"] * grad_parameters['d'+str(key)]
    return parameters


