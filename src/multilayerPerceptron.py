import numpy as np
import scipy.special

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Layer:
    def __init__(self, number_of_features, activation_units, layer_type):
        # init_epsilon = [[-sys.float_info.epsilon, sys.float_info.epsilon]]
        self.weight = (np.random.rand(activation_units, number_of_features))#  @ init_epsilon) - init_epsilon
        # self.weight = np.random.normal(0.0, pow(number_of_features, -0.5), (activation_units, number_of_features))
        # self.bias = np.ones(activation_units)
        self.bias = np.random.rand(activation_units, 1)
        self.activation_function = scipy.special.expit #sigmoid
        self.type = layer_type

    # /!\ Matrix order is very important here
    def z(self, x):
        return (self.weight @ x) +  self.bias

    def feedforward(self, x):
        self.h = self.activation_function(self.z(x))
        return self.h

class MultilayerPerceptron:
    def __init__(self, layers, lr=0.1):
        self.lr = lr
        self.layers=layers

    # def loss(self, p, y):
    #     return -((y @ np.log(p) + (1 - y) @ np.log(1 - p)) / y.shape[0])

    def forward_propagation(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.feedforward(x)
        return x

    def backward_propagation(self, inputs, targets):
        fw = self.forward_propagation(inputs)
        error = targets - fw

        t = [self.layers[0].h, inputs]
        for i, layer in enumerate(self.layers[::-1]):
            gradient = error * (layer.h * (1.0 - layer.h))
            error = layer.weight.T @ error
            layer.weight += self.lr * (gradient @ t[i].T)
            layer.bias += (self.lr * gradient)
    
