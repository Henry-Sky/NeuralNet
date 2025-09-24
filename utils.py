import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, params, grads):

        if self.m is None:
            self.m = {}
            self.v = {}
            for key, value in params:
                if key[0] == 'W' or key[0] == 'b':
                    self.m[key] = np.zeros_like(value)
                    self.v[key] = np.zeros_like(value)

        self.t += 1
        for key, value in params:
            if key[0] == 'W' or key[0] == 'b':

                print(f"{type(self.beta1)}, {type(self.m[key])}")
                a = self.beta1 * self.m[key]
                b = (1 - self.beta1) * grads[f"d{key}"]
                print(f"{a.shape}, {b.shape}")


                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[f"d{key}"]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[f"d{key}"] ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params


if __name__ == "__main__":
    params = {"W0":np.random.randn(3,2)}
    grads = {"dW0":np.random.randn(3,2)}

    # 初始化 Adam 优化器
    adam_optimizer = AdamOptimizer(0.001)

    # 更新参数
    for i in range(10):
        params = adam_optimizer.update(params, grads)
        print(f"迭代 {i + 1}: 参数 = {params}")
