import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(64, 64),
    "hidden_layer_sizes":(32*32, 16*16),
    "activation_funcs":("relu", "relu"),
    "output_func":"sigmoid",
    "loss_func":"binary_cross_entropy",
    "batch_size":64,
    "learning_rate":0.005,
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
    "binary_cross_entropy_back": lambda y_true, y_out: - (y_true / y_out) + (1 - y_true) / (1 - y_out),
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

def shuffle_data_and_labels(order_datas, order_labels):
    order_datas = np.array(order_datas)
    order_labels = np.array(order_labels)
    indices = np.random.permutation(len(order_datas))
    shuffled_datas = order_datas[indices]
    shuffled_labels = order_labels[indices]
    return shuffled_datas, shuffled_labels

def load_data(path):
    imagines = []
    img_labels = []
    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))
        regular_data = regular_img(img)
        imagines.append(regular_data)
        if filename[:3].lower() == "dog":
            img_labels.append(1)
        elif filename[:3].lower() == "cat":
            img_labels.append(0)
        else:
            raise ValueError(f"Unexpected file name: {filename}")
    shuffle_imgs, shuffle_labels = shuffle_data_and_labels(imagines, img_labels)
    data_nums = len(shuffle_imgs)
    assert data_nums == len(shuffle_labels)
    data = {"datas": shuffle_imgs, "labels": shuffle_labels, "nums": data_nums}

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

    weight_scale = 0.0001

    for index in range(hidden_layer_lens):
        parameters[f"b{index}"] = np.zeros((batch_size, hidden_layer_sizes[index]))
        if index == 0:
            parameters[f"W{index}"] = np.random.randn(w * h, hidden_layer_sizes[index]) * weight_scale

        else:
            parameters[f"W{index}"] = np.random.randn(hidden_layer_sizes[index-1], hidden_layer_sizes[index]) * weight_scale

    parameters[f"b{hidden_layer_lens}"] = np.zeros((batch_size, 1))
    parameters[f"W{hidden_layer_lens}"] = np.random.randn(hidden_layer_sizes[hidden_layer_lens-1], 1) * weight_scale

    if config["show_info"]:
        print("-----param init-----")
        for key,value in parameters.items():
            print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return parameters

def batch_forward(batch_data, parameters):
    batch_cache = {}
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)

    for index in range(hidden_layer_lens + 1):
        if index == 0:
            batch_cache["X"] = batch_data
            batch_cache[f"Z{index}"] = batch_cache["X"] @ parameters[f"W{index}"] + parameters[f"b{index}"]
        else:
            batch_cache[f"Z{index}"] = batch_cache[f"A{index-1}"] @ parameters[f"W{index}"] + parameters[f"b{index}"]
        if index == hidden_layer_lens:
            batch_cache[f"A{index}"] = activate_funcs[config["output_func"]](batch_cache[f"Z{index}"])
            batch_cache["Y"] = batch_cache[f"A{index}"]
        else:
            func = config["activation_funcs"][index]
            batch_cache[f"A{index}"] = activate_funcs[func](batch_cache[f"Z{index}"])

    if config["show_info"]:
        print("-----forward cache-----")
        for key, value in batch_cache.items():
            print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return batch_cache

def count_loss(y_true, y_out):
    func = config["loss_func"]
    return loss_funcs[func](y_true, y_out)

def batch_backward(batch_labels, parameters, batch_cache):
    grad_parameters = {}
    batch_size = config["batch_size"]
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)
    batch_labels = batch_labels.reshape(batch_size, 1)

    for index in range(hidden_layer_lens , -1, -1):
        if index == hidden_layer_lens:
            func = config["output_func"]
            loss_func = config["loss_func"]
            differ_func = activate_funcs[f"{func}_back"]
            grad_parameters[f"dA{index}"] = loss_funcs[f"{loss_func}_back"](batch_labels, batch_cache[f"Y"])
            grad_parameters[f"dZ{index}"] = grad_parameters[f"dA{index}"] * differ_func(batch_cache[f"Z{index}"])
            grad_parameters[f"dW{index}"] = batch_cache[f"A{index-1}"].T @ grad_parameters[f"dZ{index}"]
            grad_parameters[f"db{index}"] = grad_parameters[f"dA{index}"]
        elif index == 0:
            func = config["activation_funcs"][index]
            differ_func = activate_funcs[f"{func}_back"]
            grad_parameters[f"dA{index}"] = (grad_parameters[f"dA{index+1}"] * grad_parameters[f"dZ{index+1}"]) @ parameters[f"W{index+1}"].T
            grad_parameters[f"dZ{index}"] = grad_parameters[f"dA{index}"] * differ_func(batch_cache[f"Z{index}"])
            grad_parameters[f"dW{index}"] = batch_cache["X"].T @ grad_parameters[f"dZ{index}"]
            grad_parameters[f"db{index}"] = grad_parameters[f"dA{index}"]
        else:
            func = config["activation_funcs"][index]
            differ_func = activate_funcs[f"{func}_back"]
            grad_parameters[f"dA{index}"] = (grad_parameters[f"dA{index+1}"] * grad_parameters[f"dZ{index+1}"]) @ parameters[f"W{index+1}"].T
            grad_parameters[f"dZ{index}"] = grad_parameters[f"dA{index}"] * differ_func(batch_cache[f"Z{index}"])
            grad_parameters[f"dW{index}"] = batch_cache[f"A{index-1}"].T @ grad_parameters[f"dZ{index}"]
            grad_parameters[f"db{index}"] = grad_parameters[f"dA{index}"]

    if config["show_info"]:
        print("-----backward grad-----")
        for key, value in grad_parameters.items():
            if key[0:2] == "dW" or key[0:2] == "db":
                print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

    return grad_parameters

def param_update(parameters, grad):
    learn_rate = config["learning_rate"]
    for key, value in parameters.items():
        value -= learn_rate  * grad[f"d{key}"]
    return parameters

if __name__ == "__main__":
    ld = load_data(config["train_path"])
    datas = ld["datas"]
    labels = ld["labels"]
    nums = ld["nums"]
    batch_size = config["batch_size"]

    param = init_parameters()
    for iter in tqdm(range(100)):
        loss_list = []
        for i in range(0, nums-batch_size, batch_size):
            imgs = datas[i:i + batch_size]
            labs = labels[i:i + batch_size]
            cache = batch_forward(imgs, param)
            loss = count_loss(labs, cache["Y"])
            grad = batch_backward(labs, param, cache)
            param = param_update(param, grad)
            loss_list.append(loss)
        avg_loss = np.sum(loss_list) / len(loss_list)
        print(f"iter: {iter+1}, avg_loss: {avg_loss:.10f}")