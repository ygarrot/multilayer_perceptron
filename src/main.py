import argparse
import pandas as pd
import numpy as np
import sys

class Layer():
    def __init__(self, number_of_features, activation_units):
        self.init_epsilon = [-sys.float_info.epsilon, sys.float_info.epsilon]
        self.weight = np.random.rand(activation_units, number_of_features) # * init_epsilon) - init_epsilon
        self.bias = np.ones(number_of_features)
        self.h = self.weight

    # [w11 w12 w13]  * [x1 x2]
    # [w21 w22 w23]
    def z(self, x):
        return (x @ self.weight) + self.bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        self.h = self.sigmoid(self.z(x))
        return h

class MultilayerPerceptron():
    #input[x] activation1[x, ...x] activation2[x, ...x] output[2, ...x]
    def __init__(self, x):
        self.x = x
        self.lr = 0.1
        self.layers = [Layer(x.shape[0], x.shape[1]), Layer(2, x.shape[0])]

    def loss(self, p, y):
        return -((y @ np.log(p) + (1 - y) @ log(1 - p)) / y.shape[0])

    def forward_propagation(self):
        x = self.x
        for i, layer in enumerate(self.layers):
            x = layer.feedforward(x)
        return x

    def backward_propagation(self, y):
        fw = self.forward_propagation()
        error = y - x.layers[-1]
        for i in range(self.layer.length)
            delta[i] = (layer.weight.T @ delta[i + 1]) @ layer.h @ (1 - layer.h)
        # gradient descent
        for layer in enumerate(self.layers):
            layer.weight -= self.lr * delta[i]


        return 

def train(path):
    # data = pd.read_csv(path)
    data = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]])
    mp = MultilayerPerceptron(data)
    # mp.forward_propagation()
    mp.forward_propagation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()

    train(args.path)



if __name__ == '__main__':
    main()
