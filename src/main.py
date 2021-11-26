import argparse
import random
import pandas as pd
import numpy as np
import scipy.special
import sys

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer():
    def __init__(self, number_of_features, activation_units, layer_type):
        # init_epsilon = [[-sys.float_info.epsilon, sys.float_info.epsilon]]
        # self.bias = np.random.rand(activation_units, 1)
        # self.activation_function = sigmoid
        # self.weight = (np.random.rand(activation_units, number_of_features))#  @ init_epsilon) - init_epsilon
        self.weight = np.random.normal(0.0, pow(number_of_features, -0.5), (activation_units, number_of_features))
        self.bias = np.ones(activation_units)
        self.activation_function = scipy.special.expit
        self.type = layer_type

    # [w11 w12 w13] . [x1 x2]
    # [w21 w22 w23]
    def z(self, x):
        # print(self.weight, x)
        return np.dot(self.weight, x)# + self.bias

    def feedforward(self, x):
        self.h = self.activation_function(self.z(x))
        return self.h

class MultilayerPerceptron():
    def __init__(self):
        self.lr = 0.3
        self.layers = [Layer(2, 2, layer_type="input_layer"),
                       Layer(2, 1, layer_type="output_layer")]

    # def loss(self, p, y):
    #     return -((y @ np.log(p) + (1 - y) @ np.log(1 - p)) / y.shape[0])

    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.feedforward(x)
        return x

    def backward_propagation(self, x, y):
        inputs = np.array(x, ndmin=2).T
        targets = np.array(y, ndmin=1).T

        fw = self.forward_propagation(inputs)
        error = targets - fw

        t = [self.layers[0].h, inputs]
        for i, layer in enumerate(self.layers[::-1]):
            gradient = (layer.h * (1.0 - layer.h))
            e = error * gradient
            error = np.dot(layer.weight.T, error)
            layer.weight += self.lr * np.dot(e, t[i].T)
            # layer.bias += gradient

def train(path):
    # data = pd.read_csv(path)
    data = np.array([
            [0, 0, 1, 1],
            [0, 1, 0, 1]
            ])
    result = np.array([0, 1, 1, 0])
    # epoch = 5000
    epoch = 5000
    mp = MultilayerPerceptron()

    # train our perceptron
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
