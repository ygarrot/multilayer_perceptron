import argparse
import random
import pandas as pd
import numpy as np
import sys

class Layer():
    def __init__(self, number_of_features, activation_units):
        self.init_epsilon = [-sys.float_info.epsilon, sys.float_info.epsilon]
        self.weight = np.random.rand(number_of_features, activation_units) # * init_epsilon) - init_epsilon
        self.bias = np.ones(activation_units)
        # print(self.weight)
        # self.h = self.weight

    # [w11 w12 w13]  * [x1 x2]
    # [w21 w22 w23]
    def z(self, x):
        return (x @ self.weight) + self.bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        self.h = np.array(self.sigmoid(self.z(x)))
        return self.h

class MultilayerPerceptron():
    def __init__(self):
        self.lr = 0.1
        self.layers = [Layer(2, 2), Layer(2, 1)]

    def loss(self, p, y):
        return -((y @ np.log(p) + (1 - y) @ log(1 - p)) / y.shape[0])

    def forward_propagation(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.feedforward(x)
        return x

    def backward_propagation(self, x, y):
        fw = np.array(self.forward_propagation(x))
        error = np.array([y - fw])
        delta = []
        for i, layer in enumerate(self.layers[::-1]):
            d = (layer.h * (1 - layer.h))
            if (i == 0):
                e = (layer.weight @ error)
                error = e * d
            else:
                e = (layer.weight.T @ error)
                error = np.array([e * d])
            delta.append(error)

        delta = delta[::-1]
        test = np.array([np.array(x), self.layers[1].h.reshape((-1, 1))])
        for i, layer in enumerate(self.layers):
            # print("weight [{}] = {}".format(i, layer.weight))
            layer.weight += (self.lr * ((delta[i] @ test[i].T)))
            layer.weight /= layer.weight.shape[0]
            layer.bias += self.lr * (np.sum(delta[i]))
            layer.bias /= layer.h.shape[0]

def train(path):
    # data = pd.read_csv(path)
    data = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
            ])
    result = np.array([0, 1, 1, 0])
    epoch = 5000
    mp = MultilayerPerceptron()
    for _ in range(epoch):
        i = random.randint(0, len(result) - 1)
        x = [data[0][i], data[1][i]]
        y = result[i]
        # print("x, y = ", x, y)
        mp.backward_propagation(x, y)

    print(mp.forward_propagation([0, 0]))
    print(mp.forward_propagation([0, 1]))
    print(mp.forward_propagation([1, 0]))
    print(mp.forward_propagation([1, 1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()

    train(args.path)



if __name__ == '__main__':
    main()
