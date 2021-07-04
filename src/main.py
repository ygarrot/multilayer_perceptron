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
        print(self.weight)
        self.h = self.weight

    # [w11 w12 w13]  * [x1 x2]
    # [w21 w22 w23]
    def z(self, x):
        return (x @ self.weight) + self.bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        self.h = self.sigmoid(self.z(x))
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
        fw = self.forward_propagation(x)
        error = np.array([y - self.layers[-1].h])
        delta = []
        t1 = np.array([1,2])
        t2 = np.array([1])
        print(t1@t2)
        for i, layer in enumerate(self.layers[::-1]):
            error = (layer.weight.T @ error) @ (layer.h @ (1 - layer.h))
            delta.append(error)

        for layer in enumerate(self.layers):
            layer.weight -= self.lr * delta[i]
        return 

def train(path):
    # data = pd.read_csv(path)
    # data = np.array([
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4],
    #     [1, 2, 3, 4]])
    data = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
            ])
    result = np.array([0, 1, 1, 0])
    epoch = 1000
    mp = MultilayerPerceptron()
    for _ in range(1000):
        i = random.randint(0, len(result) - 1)
        x = [data[0][i], data[1][i]]
        y = result[i]
        mp.backward_propagation(x, y)

    # mp.forward_propagation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()

    train(args.path)



if __name__ == '__main__':
    main()
