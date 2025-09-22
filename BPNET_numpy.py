import os
from PIL import Image
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

config = {
    "train_path":"dogs-vs-cats/train",
    "test_path":"dogs-vs-cats/test1",
    "regular_size":(64, 64),
    "hidden_layer_sizes":(32*32, 8*8),
    "activation_funcs":("relu", "relu"),
    "output_func":"sigmoid",
    "loss_func":"binary_cross_entropy",
    "batch_size":3,
    "learning_rate":0.001,
    "show_info":True,
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
    "binary_cross_entropy_back": lambda y_true, y_out: (y_out - y_true) / y_true.size,
}

def min_max_normalize(data):
    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value) + config["epsilon"]
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
    data = {"datas": imagines, "labels": labels, "data_num":len(imagines), "label_num":len(labels)}
    assert data["data_num"] == data["label_num"]

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
    batch_size = config["batch_size"]
    assert (batch_size == batch_data.shape[0])
    hidden_layer_sizes = config["hidden_layer_sizes"]
    hidden_layer_lens = len(hidden_layer_sizes)
    (w, h) = config["regular_size"]

    for index in range(hidden_layer_lens + 1):
        if index == 0:
            cache['z'+str(index)] = batch_data @ parameters['W'+str(index)] + parameters['b'+str(index)]
        else:
            cache['z'+str(index)] = cache['a'+str(index-1)] @ parameters['W'+str(index)] + parameters['b'+str(index)]
        if index == hidden_layer_lens:
            cache['a' + str(index)] = activate_funcs[config["output_func"]](cache['z' + str(index)])
        else:
            func = config["activation_funcs"][index]
            cache['a' + str(index)] = activate_funcs[func](cache['z' + str(index)])

    if config["show_info"]:
        print("-----forward cache-----")
        for key, value in cache.items():
            print(f"{key}: {value.shape} -max:{np.max(value)} -min:{np.min(value)}")

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

if __name__ == "__main__":
    data = load_data(config["train_path"])["datas"]
    param = init_parameters()
    batch_forward(data[0:3], param)

