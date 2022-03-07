import pickle
import numpy as np
import scipy.special

def relu(x, d=False):
    if d:
        x[x >= 0] = 1
    else:
        x[x < 0] = 0
    return x

def softmax(x, d=False):
    """Compute the softmax of vector x in a numerically stable way."""
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

# def softmax(z, d=False):
#     return np.exp(z) / np.sum(np.exp(z))

def sigmoid(z, d=False):
    s = scipy.special.expit(z)
    if d:
        return z * s * (1 - s)
    return z

class Layer:
    def __init__(self, number_of_features, activation_units,
                 layer_type, activation_function=sigmoid):
        self.weight = np.random.rand(activation_units, number_of_features) * 0.01
        self.bias = np.zeros((activation_units, 1))
        self.activation_function = activation_function
        self.type = layer_type
        assert(self.weight.shape == (activation_units, number_of_features))
        assert(self.bias.shape == (activation_units, 1))

    def linear_forward(self, x):
        self.z = (self.weight @ x) + self.bias
        assert(self.z.shape == (self.weight.shape[0], x.shape[1]))
        return self.z

    def linear_backward(self, A_prev):
        dZ = self.activation_function(A_prev, d=True)
        m = A_prev.shape[0]
        dW = np.dot(dZ, self.A_prev.T) / m
        db = np.sum(np.matrix(dZ), axis=1) / m
        dA_prev = np.dot(self.weight.T, dZ)
        assert(dA_prev.shape == self.A_prev.shape)
        assert(dW.shape == self.weight.shape)
        assert (db.shape == self.bias.shape)
        return dA_prev, dW, db

    def update_parameters(self, rate, dW, db):
        self.weight -= rate * dW
        self.bias -= rate * db

    def linear_activation_forward(self, A_prev):
        self.A_prev = A_prev
        self.A = self.activation_function(self.linear_forward(A_prev))
        assert (self.A.shape == (self.weight.shape[0], A_prev.shape[1]))
        return self.A

class MultilayerPerceptron:
    def __init__(self, layers, lr=0.1, path="mp.pickle"):
        self.lr = lr
        self.layers = layers
        self.path = path

    def loss(self, p, y, eps=1e-15):
        m = y.shape[0]
        return np.squeeze(np.sum(y * np.log(p+eps) + (1 - y) * np.log(1 - p+eps)) / -m)

    def forward_propagation(self, x):
        for layer in self.layers:
            x = layer.linear_activation_forward(x)
        return x

    def backward_propagation(self, X, Y):
        AL = self.forward_propagation(X)
        Y = Y.reshape(AL.shape)
        A_prev = AL-Y
        for i in range(len(self.layers)-1, -1, -1):
            A_prev, dW, db = self.layers[i].linear_backward(A_prev)
            self.layers[i].update_parameters(self.lr, dW, db)
        return AL

    def load(self):
        with open(self.path, 'rb') as handle:
            self = pickle.load(handle)

    def save(self):
        with open(self.path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 

