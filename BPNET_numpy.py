import os
from tqdm import tqdm
import logging
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(64, 64),
    "hidden_layer_sizes":(32*32, 16*16, 8*8),
    "activation_funcs":("relu", "relu", "relu"),
    "output_func":"sigmoid",
    "loss_func":"binary_cross_entropy",
    "batch_size":24,
    "learning_rate":0.01,
    "show_info":False,
    "epsilon":1e-6,
}

activate_funcs = {
    "relu": lambda x: np.maximum(x, 0),
    "relu_back": lambda x: np.where(x > 0, 1, 0),
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "sigmoid_back": lambda  x: np.exp(-x) / (1 + np.exp(-x))**2,
}

loss_funcs = {
    "binary_cross_entropy": lambda y_true, y_out: - np.mean(y_true * np.log(y_out) + (1 - y_true) * np.log(1 - y_out)),
    "binary_cross_entropy_back": lambda y_true, y_out: np.mean(y_out - y_true),
}

def min_max_normalize(origin_data):
    min_value = np.min(origin_data)
    max_value = np.max(origin_data)
    normalized_data = (origin_data - min_value) / (max_value - min_value) + config["epsilon"]
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
    imagines = np.array(imagines)
    labels = np.array(labels)
    nums = len(imagines)
    assert nums == len(labels)
    data = {"datas": imagines, "labels": labels, "nums": nums}

    if config["show_info"]:
        print("-----data load-----")
        print(f"imagines info -max: {np.max(imagines)}, -min: {np.min(imagines)}")

    return data

def init_parameters():
    parameters = {}
    batch_size = config["batch_size"]
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)
    (w, h) = config["regular_size"]

    for index in range(hidden_layer_lens):
        parameters['b' + str(index)] = np.zeros((batch_size, hidden_layer_sizes[index]))
        if index == 0:
            parameters['W'+str(index)] = np.random.randn(w * h, hidden_layer_sizes[index]) * 0.01

        else:
            parameters['W'+str(index)] = np.random.randn(hidden_layer_sizes[index-1], hidden_layer_sizes[index]) * 0.01

    parameters['b'+str(hidden_layer_lens)] = np.zeros((batch_size, 1))
    parameters['W'+str(hidden_layer_lens)] = np.random.randn(hidden_layer_sizes[hidden_layer_lens-1], 1) * 0.01

    if config["show_info"]:
        print("-----param init-----")
        for key,value in parameters.items():
            print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return parameters

def batch_forward(batch_data, parameters):
    cache = {}
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)

    for index in range(hidden_layer_lens + 1):
        if index == 0:
            cache[f"A{index-1}"] = batch_data
            cache[f"Z{index}"] = cache[f"A{index-1}"] @ parameters[f"W{index}"] + parameters[f"b{index}"]
        else:
            cache[f"Z{index}"] = cache[f"A{index-1}"] @ parameters[f"W{index}"] + parameters[f"b{index}"]
        if index == hidden_layer_lens:
            cache[f"A{index}"] = activate_funcs[config["output_func"]](cache[f"Z{index}"])
            cache["Y"] = cache[f"A{index}"]
        else:
            func = config["activation_funcs"][index]
            cache[f"A{index}"] = activate_funcs[func](cache[f"Z{index}"])

    if config["show_info"]:
        print("-----forward cache-----")
        for key, value in cache.items():
            print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return cache

def count_loss(y_true, y_out):
    func = config["loss_func"]
    return loss_funcs[func](y_true, y_out)

def batch_backward(labels, parameters, cache):
    grad_parameters = {}
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)

    for index in range(hidden_layer_lens , -1, -1):
        if index == hidden_layer_lens:
            func = config["output_func"]
            loss_func = config["loss_func"]
            differ_func = activate_funcs[f"{func}_back"]
            grad_parameters[f"dA{index}"] = loss_funcs[f"{loss_func}_back"](labels, cache[f"A{index}"])
            grad_parameters[f"dZ{index}"] = grad_parameters[f"dA{index}"] * differ_func(cache[f"Z{index}"])
            grad_parameters[f"dW{index}"] = cache[f"A{index-1}"].T @ grad_parameters[f"dZ{index}"]
            grad_parameters[f"db{index}"] = grad_parameters[f"dA{index}"]
        else:
            func = config["activation_funcs"][index]
            differ_func = activate_funcs[f"{func}_back"]
            grad_parameters[f"dA{index}"] = (grad_parameters[f"dA{index+1}"] * grad_parameters[f"dZ{index+1}"]) @ parameters[f"W{index+1}"].T
            grad_parameters[f"dZ{index}"] = grad_parameters[f"dA{index}"] * differ_func(cache[f"Z{index}"])
            grad_parameters[f"dW{index}"] = cache[f"A{index-1}"].T @ grad_parameters[f"dZ{index}"]
            grad_parameters[f"db{index}"] = grad_parameters[f"dA{index}"]

    if config["show_info"]:
        print("-----backward grad-----")
        for key, value in grad_parameters.items():
            if key[0:2] == "dW" or key[0:2] == "db":
                print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return grad_parameters

def param_update(param, grad):
    learn_rate = config["learning_rate"]

    for key, value in param.items():
        value -= learn_rate  * grad[f"d{key}"]

    return param

if __name__ == "__main__":
    ld = load_data(config["train_path"])
    data = ld["datas"]
    label = ld["labels"]
    nums = ld["nums"]
    batch_size = config["batch_size"]

    param = init_parameters()
    for iter in tqdm(range(10)):
        loss_list = []
        for i in range(0, nums-batch_size, batch_size):
            imgs = data[i:i+batch_size]
            labs = label[i:i + batch_size]
            cache = batch_forward(imgs, param)
            loss = count_loss(labs, cache["Y"])
            grad = batch_backward(labs, param, cache)
            param = param_update(param, grad)
            loss_list.append(loss)
        avg_loss = np.sum(loss_list) / len(loss_list)
        print(f"iter: {iter+1}, avg_loss: {avg_loss}")


