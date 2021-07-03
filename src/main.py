import argparse
import pandas as pd
import numpy as np
import sys

class Layer():
    def __init__(self, number_of_features, activation_units):
        self.init_epsilon = [-sys.float_info.epsilon, sys.float_info.epsilon]
        self.weight = np.random.rand(activation_units, number_of_features) # * init_epsilon) - init_epsilon
        self.bias = np.ones(number_of_features)

    # [w11 w12 w13]  * [x1 x2]
    # [w21 w22 w23]
    def z(self, x):
        print(x.shape)
        print(self.weight.shape)
        print(self.weight)
        return (x @ self.weight) + self.bias

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, x):
        return self.sigmoid(self.z(x))

    def backpropagation(activation, y):
        for i in range(activation, 0):
            delta = activation - y

class MultilayerPerceptron():
    #input[x] activation1[x, ...x] activation2[x, ...x] output[2, ...x]
    def __init__(self, x):
        self.x = x
        self.layers = [Layer(x.shape[0], x.shape[1]), Layer(2, x.shape[0])]


    def forward_propagation(self):
        # activation = [x, a2, a3 ... ]
        x = self.x
        for i, layer in enumerate(self.layers):
            # print(x.shape)
            # print(x)
            x = layer.feedforward(x)
            # print(x)

def train(path):
    # data = pd.read_csv(path)
    data = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]])
    mp = MultilayerPerceptron(data)
    mp.forward_propagation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()

    train(args.path)



if __name__ == '__main__':
    main()
