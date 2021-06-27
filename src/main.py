import argparse
import pandas as pd
import numpy as np
import sys

class Layer():
    def __init__(self, activation_units, number_of_features):
        init_epsilon = [-sys.float_info.epsilon, sys.float_info.epsilon]
        weight = np.random.rand(number_of_features, activation_units) # * init_epsilon) - init_epsilon
        bias = np.ones(activation_units)
        print(weight)
        print(weight.shape)

    # [w11 w12 w13]  * [x1 x2]
    # [w21 w22 w23]
    def z(self, x, bias):
        return (x @ self.weight) + self.bias

    def g(self, x):
        return g(x)

    def backpropagation(activation, y):
        for i in range(activation, 0):
            delta = activation - y

class MultilayerPerceptron():
    #input[x] activation1[x, ...x] activation2[x, ...x] output[2, ...x]
    def __init__(self, x):
        layers = [Layer(0, 0), Layer(x.shape[0], x.shape[1]), Layer(x.shape[0], 2)]


    def forward_propagation(self, x, weight):
        # activation = [x, a2, a3 ... ]
        for i, layer in enumerate(self.layers):
            activation[i+1] = layer.g(activation[i])
    


def train(path):
    data = pd.read_csv(path)
    mp = MultilayerPerceptron(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, default="./resources/data.csv", help="dataset_train.csv")
    args = parser.parse_args()

    train(args.path)



if __name__ == '__main__':
    main()
